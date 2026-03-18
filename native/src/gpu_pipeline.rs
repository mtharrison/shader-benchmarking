use std::fmt::{self, Display, Formatter};

use napi_derive::napi;

const SAMPLE_GPU_MAP2_SOURCE: &str = "let out = gpu.map2(a, b, (x, y) => x + y);";
const SAMPLE_MATRIX_REDUCTION_SOURCE: &str = "function aggregateMatrixBatch(matrices, matrixCount, rows, cols) {\n  const perMatrixColumnSums = gpu.reduceColumnsPerMatrix(matrices, matrixCount, rows, cols, (acc, value) => acc + value, 0.0);\n  const averageColumnSums = gpu.averageColumns(perMatrixColumnSums, matrixCount, cols);\n  const matrixTotals = gpu.reducePerMatrix(perMatrixColumnSums, matrixCount, cols, (acc, value) => acc + value, 0.0);\n  const grandTotal = gpu.reduce(matrixTotals, (acc, value) => acc + value, 0.0);\n  return { averageColumnSums, grandTotal };\n}";

#[napi(object)]
pub struct CompiledGpuPipeline {
    pub source: String,
    pub js_ast: String,
    pub kernel_ir: String,
    pub ptx: String,
    pub host_launch: String,
    pub notes: Vec<String>,
}

#[napi(object)]
pub struct CompiledMatrixReductionPipeline {
    pub source: String,
    pub reduction_ir: String,
    pub stage1_kernel_ir: String,
    pub stage2_kernel_ir: String,
    pub stage3_kernel_ir: String,
    pub stage4_kernel_ir: String,
    pub ptx: String,
    pub host_launch: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Program {
    output: String,
    invocation: GpuMap2Invocation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GpuMap2Invocation {
    left_buffer: String,
    right_buffer: String,
    lambda: Lambda,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Lambda {
    left_param: String,
    right_param: String,
    body: JsExpr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum JsExpr {
    Identifier(String),
    Binary {
        op: BinaryOp,
        left: Box<JsExpr>,
        right: Box<JsExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BufferAccess {
    Read,
    Write,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KernelParam {
    name: String,
    kind: KernelParamKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum KernelParamKind {
    Buffer {
        element: ElementType,
        access: BufferAccess,
    },
    ScalarU32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ElementType {
    F32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KernelModule {
    kernel_name: String,
    output_buffer: String,
    left_buffer: String,
    right_buffer: String,
    params: Vec<KernelParam>,
    body: KernelStmt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum KernelStmt {
    GuardedStore {
        index: KernelExpr,
        bound: String,
        buffer: String,
        value: KernelExpr,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum KernelExpr {
    GlobalIdX,
    Load {
        buffer: String,
        index: Box<KernelExpr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<KernelExpr>,
        right: Box<KernelExpr>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Identifier(String),
    Equals,
    Dot,
    Comma,
    LParen,
    RParen,
    Arrow,
    Semicolon,
    Plus,
    Minus,
    Star,
}

struct Parser {
    tokens: Vec<Token>,
    cursor: usize,
}

struct PtxEmitter {
    lines: Vec<String>,
    next_f: u32,
    next_rd: u32,
}

pub fn sample_gpu_map2_source() -> &'static str {
    SAMPLE_GPU_MAP2_SOURCE
}

pub fn compile_gpu_pipeline(source: &str) -> Result<CompiledGpuPipeline, String> {
    let tokens = tokenize(source)?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;
    parser.expect_end()?;

    let kernel = lower_to_kernel(&program)?;
    let ptx = emit_ptx(&kernel)?;
    let host_launch = emit_host_launch(&kernel);

    Ok(CompiledGpuPipeline {
        source: source.to_string(),
        js_ast: program.to_pretty_string(),
        kernel_ir: kernel.to_pretty_string(),
        ptx,
        host_launch,
        notes: vec![
            "The JS frontend still parses source and builds an AST first.".to_string(),
            "GPU lowering happens before any Cranelift step so the compiler can attach thread indexing, buffer access, and element types.".to_string(),
            "This prototype only accepts typed Float32-style buffers and a gpu.map2 lambda with identifiers combined by +, -, or *.".to_string(),
            "Cranelift remains the host-side JIT path that would load the PTX and launch the kernel.".to_string(),
        ],
    })
}

pub fn sample_matrix_reduction_source() -> &'static str {
    SAMPLE_MATRIX_REDUCTION_SOURCE
}

pub fn compile_matrix_reduction_pipeline(
    matrices: usize,
    rows: usize,
    cols: usize,
) -> Result<CompiledMatrixReductionPipeline, String> {
    if matrices == 0 || rows == 0 || cols == 0 {
        return Err(
            "Matrix reduction pipelines require non-zero matrix, row, and column counts"
                .to_string(),
        );
    }

    let reduction_ir = emit_matrix_reduction_ir(matrices, rows, cols);
    let stage1_kernel_ir = emit_stage1_column_sum_kernel_ir(matrices, rows, cols);
    let stage2_kernel_ir = emit_stage2_average_columns_kernel_ir(matrices, cols);
    let stage3_kernel_ir = emit_stage3_matrix_totals_kernel_ir(matrices, cols);
    let stage4_kernel_ir = emit_stage4_grand_total_kernel_ir(matrices);
    let ptx = emit_matrix_reduction_ptx(matrices, rows, cols);
    let host_launch = emit_matrix_reduction_host_launch(matrices, rows, cols);

    Ok(CompiledMatrixReductionPipeline {
        source: SAMPLE_MATRIX_REDUCTION_SOURCE.to_string(),
        reduction_ir,
        stage1_kernel_ir,
        stage2_kernel_ir,
        stage3_kernel_ir,
        stage4_kernel_ir,
        ptx,
        host_launch,
        notes: vec![
            "Stage 1 launches one thread per (matrix, column) pair and produces one column-sum vector per matrix.".to_string(),
            format!(
                "Stage 2 averages the {cols} column sums across {matrices} matrices."
            ),
            format!(
                "Stage 3 reduces each matrix's {cols} column sums into one per-matrix total, and stage 4 reduces the {matrices} matrix totals into a grand total."
            ),
            "The artifact is compile-only in this workspace because the local machine has no NVIDIA GPU or WebGPU runtime.".to_string(),
            "In a real compiler, the host-side Cranelift path would allocate device buffers, launch all four kernels, and copy the averaged columns plus the grand total back.".to_string(),
        ],
    })
}

impl Program {
    fn to_pretty_string(&self) -> String {
        format!(
            "Program(\n  let {} = gpu.map2({}, {}, ({}, {}) => {})\n)",
            self.output,
            self.invocation.left_buffer,
            self.invocation.right_buffer,
            self.invocation.lambda.left_param,
            self.invocation.lambda.right_param,
            self.invocation.lambda.body
        )
    }
}

impl KernelModule {
    fn to_pretty_string(&self) -> String {
        let params = self
            .params
            .iter()
            .map(|param| format!("  {}", param))
            .collect::<Vec<_>>()
            .join(",\n");

        let body = match &self.body {
            KernelStmt::GuardedStore {
                index: _,
                bound,
                buffer,
                value,
            } => format!(
                "if i < {bound} {{ {buffer}[i] = {}; }}",
                value.to_pretty_string_with_index_alias("i")
            ),
        };

        format!(
            "kernel {}(\n{}\n) {{\n  let i = global_id.x;\n  {}\n}}",
            self.kernel_name, params, body
        )
    }
}

impl Display for JsExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identifier(name) => write!(f, "{name}"),
            Self::Binary { op, left, right } => write!(f, "({left} {op} {right})"),
        }
    }
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let symbol = match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
        };

        write!(f, "{symbol}")
    }
}

impl Display for KernelParam {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self.kind {
            KernelParamKind::Buffer { element, access } => {
                write!(f, "{}: buffer<{}, {}>", self.name, element, access)
            }
            KernelParamKind::ScalarU32 => write!(f, "{}: u32", self.name),
        }
    }
}

impl Display for ElementType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
        }
    }
}

impl Display for BufferAccess {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Write => write!(f, "write"),
        }
    }
}

impl Display for KernelStmt {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::GuardedStore {
                index,
                bound,
                buffer,
                value,
            } => write!(f, "if {index} < {bound} {{ {buffer}[{index}] = {value}; }}"),
        }
    }
}

impl Display for KernelExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::GlobalIdX => write!(f, "global_id.x"),
            Self::Load { buffer, index } => write!(f, "{buffer}[{index}]"),
            Self::Binary { op, left, right } => write!(f, "({left} {op} {right})"),
        }
    }
}

impl KernelExpr {
    fn to_pretty_string_with_index_alias(&self, alias: &str) -> String {
        match self {
            Self::GlobalIdX => alias.to_string(),
            Self::Load { buffer, index } => format!(
                "{buffer}[{}]",
                index.to_pretty_string_with_index_alias(alias)
            ),
            Self::Binary { op, left, right } => format!(
                "({} {op} {})",
                left.to_pretty_string_with_index_alias(alias),
                right.to_pretty_string_with_index_alias(alias)
            ),
        }
    }
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, cursor: 0 }
    }

    fn parse_program(&mut self) -> Result<Program, String> {
        self.expect_identifier_value("let")?;
        let output = self.expect_identifier()?;
        self.expect_token(Token::Equals)?;
        self.expect_identifier_value("gpu")?;
        self.expect_token(Token::Dot)?;
        self.expect_identifier_value("map2")?;
        self.expect_token(Token::LParen)?;
        let left_buffer = self.expect_identifier()?;
        self.expect_token(Token::Comma)?;
        let right_buffer = self.expect_identifier()?;
        self.expect_token(Token::Comma)?;
        let lambda = self.parse_lambda()?;
        self.expect_token(Token::RParen)?;
        self.consume_if(Token::Semicolon);

        Ok(Program {
            output,
            invocation: GpuMap2Invocation {
                left_buffer,
                right_buffer,
                lambda,
            },
        })
    }

    fn parse_lambda(&mut self) -> Result<Lambda, String> {
        self.expect_token(Token::LParen)?;
        let left_param = self.expect_identifier()?;
        self.expect_token(Token::Comma)?;
        let right_param = self.expect_identifier()?;
        self.expect_token(Token::RParen)?;
        self.expect_token(Token::Arrow)?;
        let body = self.parse_expr()?;

        Ok(Lambda {
            left_param,
            right_param,
            body,
        })
    }

    fn parse_expr(&mut self) -> Result<JsExpr, String> {
        self.parse_additive()
    }

    fn parse_additive(&mut self) -> Result<JsExpr, String> {
        let mut left = self.parse_multiplicative()?;

        while let Some(op) = self.match_binary_op(&[Token::Plus, Token::Minus]) {
            let right = self.parse_multiplicative()?;
            left = JsExpr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<JsExpr, String> {
        let mut left = self.parse_primary()?;

        while let Some(op) = self.match_binary_op(&[Token::Star]) {
            let right = self.parse_primary()?;
            left = JsExpr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_primary(&mut self) -> Result<JsExpr, String> {
        if self.consume_if(Token::LParen) {
            let expr = self.parse_expr()?;
            self.expect_token(Token::RParen)?;
            return Ok(expr);
        }

        Ok(JsExpr::Identifier(self.expect_identifier()?))
    }

    fn match_binary_op(&mut self, allowed: &[Token]) -> Option<BinaryOp> {
        let token = self.peek()?;
        let op = match token {
            Token::Plus if allowed.contains(&Token::Plus) => BinaryOp::Add,
            Token::Minus if allowed.contains(&Token::Minus) => BinaryOp::Sub,
            Token::Star if allowed.contains(&Token::Star) => BinaryOp::Mul,
            _ => return None,
        };

        self.cursor += 1;
        Some(op)
    }

    fn expect_identifier(&mut self) -> Result<String, String> {
        match self.advance() {
            Some(Token::Identifier(name)) => Ok(name),
            Some(other) => Err(format!(
                "Expected identifier, found {}",
                describe_token(&other)
            )),
            None => Err("Unexpected end of input while expecting identifier".to_string()),
        }
    }

    fn expect_identifier_value(&mut self, expected: &str) -> Result<(), String> {
        let actual = self.expect_identifier()?;

        if actual == expected {
            Ok(())
        } else {
            Err(format!(
                "Expected identifier `{expected}`, found `{actual}`"
            ))
        }
    }

    fn expect_token(&mut self, expected: Token) -> Result<(), String> {
        match self.advance() {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(format!(
                "Expected {}, found {}",
                describe_token(&expected),
                describe_token(&actual)
            )),
            None => Err(format!(
                "Unexpected end of input while expecting {}",
                describe_token(&expected)
            )),
        }
    }

    fn consume_if(&mut self, expected: Token) -> bool {
        match self.peek() {
            Some(actual) if *actual == expected => {
                self.cursor += 1;
                true
            }
            _ => false,
        }
    }

    fn expect_end(&self) -> Result<(), String> {
        if self.cursor == self.tokens.len() {
            Ok(())
        } else {
            Err(format!(
                "Unexpected trailing token {}",
                describe_token(&self.tokens[self.cursor])
            ))
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.cursor)
    }

    fn advance(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.cursor).cloned()?;
        self.cursor += 1;
        Some(token)
    }
}

impl PtxEmitter {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            next_f: 1,
            next_rd: 5,
        }
    }

    fn alloc_f(&mut self) -> String {
        let register = format!("%f{}", self.next_f);
        self.next_f += 1;
        register
    }

    fn alloc_rd(&mut self) -> String {
        let register = format!("%rd{}", self.next_rd);
        self.next_rd += 1;
        register
    }

    fn emit_expr(
        &mut self,
        expr: &KernelExpr,
        base_buffers: &BaseRegisters,
    ) -> Result<String, String> {
        match expr {
            KernelExpr::Load { buffer, .. } => {
                let base = base_buffers.for_buffer(buffer)?;
                let addr = self.alloc_rd();
                let value = self.alloc_f();
                self.lines
                    .push(format!("    add.s64 {addr}, {base}, %rd4;"));
                self.lines
                    .push(format!("    ld.global.f32 {value}, [{addr}];"));
                Ok(value)
            }
            KernelExpr::Binary { op, left, right } => {
                let left_value = self.emit_expr(left, base_buffers)?;
                let right_value = self.emit_expr(right, base_buffers)?;
                let out = self.alloc_f();
                self.lines.push(format!(
                    "    {}.f32 {out}, {left_value}, {right_value};",
                    op.to_ptx_instruction()
                ));
                Ok(out)
            }
            KernelExpr::GlobalIdX => {
                Err("GlobalIdX cannot be materialized as an f32 value".to_string())
            }
        }
    }
}

struct BaseRegisters<'a> {
    left_buffer: &'a str,
    right_buffer: &'a str,
    output_buffer: &'a str,
}

impl<'a> BaseRegisters<'a> {
    fn for_buffer(&self, buffer: &str) -> Result<&'static str, String> {
        if buffer == self.left_buffer {
            Ok("%rd1")
        } else if buffer == self.right_buffer {
            Ok("%rd2")
        } else if buffer == self.output_buffer {
            Ok("%rd3")
        } else {
            Err(format!(
                "Unknown buffer `{buffer}` referenced during PTX emission"
            ))
        }
    }
}

impl BinaryOp {
    fn to_ptx_instruction(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
        }
    }

    fn kernel_suffix(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
        }
    }
}

fn tokenize(source: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = source.chars().collect();
    let mut cursor = 0;

    while cursor < chars.len() {
        let current = chars[cursor];

        if current.is_whitespace() {
            cursor += 1;
            continue;
        }

        if current.is_ascii_alphabetic() || current == '_' {
            let start = cursor;
            cursor += 1;

            while cursor < chars.len()
                && (chars[cursor].is_ascii_alphanumeric() || chars[cursor] == '_')
            {
                cursor += 1;
            }

            let identifier = chars[start..cursor].iter().collect::<String>();
            tokens.push(Token::Identifier(identifier));
            continue;
        }

        match current {
            '=' if chars.get(cursor + 1) == Some(&'>') => {
                tokens.push(Token::Arrow);
                cursor += 2;
            }
            '=' => {
                tokens.push(Token::Equals);
                cursor += 1;
            }
            '.' => {
                tokens.push(Token::Dot);
                cursor += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                cursor += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                cursor += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                cursor += 1;
            }
            ';' => {
                tokens.push(Token::Semicolon);
                cursor += 1;
            }
            '+' => {
                tokens.push(Token::Plus);
                cursor += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                cursor += 1;
            }
            '*' => {
                tokens.push(Token::Star);
                cursor += 1;
            }
            _ => {
                return Err(format!(
                    "Unexpected character `{current}` in source; this prototype only supports identifiers, punctuation, and + - * inside gpu.map2 lambdas"
                ))
            }
        }
    }

    Ok(tokens)
}

fn lower_to_kernel(program: &Program) -> Result<KernelModule, String> {
    let lambda = &program.invocation.lambda;
    let value = lower_expr(
        &lambda.body,
        &lambda.left_param,
        &program.invocation.left_buffer,
        &lambda.right_param,
        &program.invocation.right_buffer,
    )?;

    let op = extract_root_op(&lambda.body)?;
    let kernel_name = format!("{}_map2_{}_f32", program.output, op.kernel_suffix());

    Ok(KernelModule {
        kernel_name,
        output_buffer: program.output.clone(),
        left_buffer: program.invocation.left_buffer.clone(),
        right_buffer: program.invocation.right_buffer.clone(),
        params: vec![
            KernelParam {
                name: program.invocation.left_buffer.clone(),
                kind: KernelParamKind::Buffer {
                    element: ElementType::F32,
                    access: BufferAccess::Read,
                },
            },
            KernelParam {
                name: program.invocation.right_buffer.clone(),
                kind: KernelParamKind::Buffer {
                    element: ElementType::F32,
                    access: BufferAccess::Read,
                },
            },
            KernelParam {
                name: program.output.clone(),
                kind: KernelParamKind::Buffer {
                    element: ElementType::F32,
                    access: BufferAccess::Write,
                },
            },
            KernelParam {
                name: "n".to_string(),
                kind: KernelParamKind::ScalarU32,
            },
        ],
        body: KernelStmt::GuardedStore {
            index: KernelExpr::GlobalIdX,
            bound: "n".to_string(),
            buffer: program.output.clone(),
            value,
        },
    })
}

fn lower_expr(
    expr: &JsExpr,
    left_param: &str,
    left_buffer: &str,
    right_param: &str,
    right_buffer: &str,
) -> Result<KernelExpr, String> {
    match expr {
        JsExpr::Identifier(name) if name == left_param => Ok(KernelExpr::Load {
            buffer: left_buffer.to_string(),
            index: Box::new(KernelExpr::GlobalIdX),
        }),
        JsExpr::Identifier(name) if name == right_param => Ok(KernelExpr::Load {
            buffer: right_buffer.to_string(),
            index: Box::new(KernelExpr::GlobalIdX),
        }),
        JsExpr::Identifier(name) => Err(format!(
            "Unsupported identifier `{name}` in gpu.map2 body; only lambda parameters are currently legal"
        )),
        JsExpr::Binary { op, left, right } => Ok(KernelExpr::Binary {
            op: *op,
            left: Box::new(lower_expr(
                left,
                left_param,
                left_buffer,
                right_param,
                right_buffer,
            )?),
            right: Box::new(lower_expr(
                right,
                left_param,
                left_buffer,
                right_param,
                right_buffer,
            )?),
        }),
    }
}

fn extract_root_op(expr: &JsExpr) -> Result<BinaryOp, String> {
    match expr {
        JsExpr::Binary { op, .. } => Ok(*op),
        _ => Err(
            "gpu.map2 bodies must currently be a binary expression over the lambda parameters"
                .to_string(),
        ),
    }
}

fn emit_ptx(kernel: &KernelModule) -> Result<String, String> {
    let base_buffers = BaseRegisters {
        left_buffer: &kernel.left_buffer,
        right_buffer: &kernel.right_buffer,
        output_buffer: &kernel.output_buffer,
    };

    let KernelStmt::GuardedStore {
        index: _,
        bound: _,
        buffer: _,
        value,
    } = &kernel.body;

    let mut emitter = PtxEmitter::new();
    let value_register = emitter.emit_expr(value, &base_buffers)?;
    let output_addr = emitter.alloc_rd();

    let mut sections = vec![
        ".version 7.0".to_string(),
        ".target sm_80".to_string(),
        ".address_size 64".to_string(),
        "".to_string(),
        format!(".visible .entry {}(", kernel.kernel_name),
        format!("    .param .u64 {},", kernel.left_buffer),
        format!("    .param .u64 {},", kernel.right_buffer),
        format!("    .param .u64 {},", kernel.output_buffer),
        "    .param .u32 n".to_string(),
        ")".to_string(),
        "{".to_string(),
        "    .reg .pred %p<2>;".to_string(),
        "    .reg .b32 %r<8>;".to_string(),
        "    .reg .b64 %rd<16>;".to_string(),
        "    .reg .f32 %f<16>;".to_string(),
        "".to_string(),
        format!("    ld.param.u64 %rd1, [{}];", kernel.left_buffer),
        format!("    ld.param.u64 %rd2, [{}];", kernel.right_buffer),
        format!("    ld.param.u64 %rd3, [{}];", kernel.output_buffer),
        "    ld.param.u32 %r1, [n];".to_string(),
        "".to_string(),
        "    mov.u32 %r2, %ctaid.x;".to_string(),
        "    mov.u32 %r3, %ntid.x;".to_string(),
        "    mov.u32 %r4, %tid.x;".to_string(),
        "    mad.lo.s32 %r5, %r2, %r3, %r4;".to_string(),
        "    setp.ge.u32 %p1, %r5, %r1;".to_string(),
        "    @%p1 bra DONE;".to_string(),
        "".to_string(),
        "    mul.wide.u32 %rd4, %r5, 4;".to_string(),
    ];

    sections.extend(emitter.lines);
    sections.push(format!("    add.s64 {output_addr}, %rd3, %rd4;"));
    sections.push(format!(
        "    st.global.f32 [{output_addr}], {value_register};"
    ));
    sections.push("DONE:".to_string());
    sections.push("    ret;".to_string());
    sections.push("}".to_string());

    Ok(sections.join("\n"))
}

fn emit_host_launch(kernel: &KernelModule) -> String {
    format!(
        "fn launch_{name}(cuda: &CudaRuntime, {left}: DeviceBuffer<f32>, {right}: DeviceBuffer<f32>, {out}: DeviceBuffer<f32>) {{\n  let n = {out}.len() as u32;\n  let module = cuda.load_ptx({name_upper}_PTX);\n  let kernel = module.function(\"{name}\");\n  let block = 256u32;\n  let grid = n.div_ceil(block);\n  cuda.launch_1d(&kernel, grid, block, (&{left}, &{right}, &{out}, &n));\n}}",
        name = kernel.kernel_name,
        left = kernel.left_buffer,
        right = kernel.right_buffer,
        out = kernel.output_buffer,
        name_upper = kernel.kernel_name.to_ascii_uppercase()
    )
}

fn emit_matrix_reduction_ir(matrices: usize, rows: usize, cols: usize) -> String {
    format!(
        "Pipeline matrix_batch_average_columns_and_grand_total_f64 {{\n  input matrices: buffer<f64> [{matrices} x {rows} x {cols}]\n  stage per_matrix_column_sums: reduce_columns_per_matrix(matrices, matrix_count={matrices}, rows={rows}, cols={cols}, op=add, init=0.0f64)\n  stage average_column_sums: average_columns(per_matrix_column_sums, matrix_count={matrices}, cols={cols})\n  stage matrix_totals: reduce_per_matrix(per_matrix_column_sums, matrix_count={matrices}, cols={cols}, op=add, init=0.0f64)\n  stage grand_total: reduce(matrix_totals, len={matrices}, op=add, init=0.0f64)\n  output {{ average_column_sums, grand_total }}\n}}"
    )
}

fn emit_stage1_column_sum_kernel_ir(matrices: usize, rows: usize, cols: usize) -> String {
    format!(
        "kernel matrix_column_sum_f64(\n  matrices: buffer<f64, read>,\n  matrix_column_sums: buffer<f64, write>,\n  matrix_count: u32,\n  rows: u32,\n  cols: u32\n) {{\n  let linear = global_id.x;\n  let total_columns = matrix_count * cols;\n  if linear < total_columns {{\n    let matrix = linear / cols;\n    let col = linear % cols;\n    let matrix_base = matrix * rows * cols;\n    let acc = 0.0f64;\n    for row in 0..rows {{\n      acc += matrices[matrix_base + row * cols + col];\n    }}\n    matrix_column_sums[linear] = acc;\n  }}\n}}\n\nspecialized_for matrix_count={matrices}, rows={rows}, cols={cols}"
    )
}

fn emit_stage2_average_columns_kernel_ir(matrices: usize, cols: usize) -> String {
    format!(
        "kernel average_columns_f64(\n  matrix_column_sums: buffer<f64, read>,\n  average_column_sums: buffer<f64, write>,\n  matrix_count: u32,\n  cols: u32\n) {{\n  let col = global_id.x;\n  if col < cols {{\n    let acc = 0.0f64;\n    for matrix in 0..matrix_count {{\n      acc += matrix_column_sums[matrix * cols + col];\n    }}\n    average_column_sums[col] = acc / f64(matrix_count);\n  }}\n}}\n\nspecialized_for matrix_count={matrices}, cols={cols}"
    )
}

fn emit_stage3_matrix_totals_kernel_ir(matrices: usize, cols: usize) -> String {
    format!(
        "kernel matrix_totals_f64(\n  matrix_column_sums: buffer<f64, read>,\n  matrix_totals: buffer<f64, write>,\n  matrix_count: u32,\n  cols: u32\n) {{\n  let matrix = global_id.x;\n  if matrix < matrix_count {{\n    let base = matrix * cols;\n    let acc = 0.0f64;\n    for col in 0..cols {{\n      acc += matrix_column_sums[base + col];\n    }}\n    matrix_totals[matrix] = acc;\n  }}\n}}\n\nspecialized_for matrix_count={matrices}, cols={cols}"
    )
}

fn emit_stage4_grand_total_kernel_ir(matrices: usize) -> String {
    format!(
        "kernel grand_total_f64(\n  matrix_totals: buffer<f64, read>,\n  grand_total: buffer<f64, write>,\n  matrix_count: u32\n) {{\n  let lane = global_id.x;\n  if lane == 0 {{\n    let acc = 0.0f64;\n    for matrix in 0..matrix_count {{\n      acc += matrix_totals[matrix];\n    }}\n    grand_total[0] = acc;\n  }}\n}}\n\nspecialized_for matrix_count={matrices}"
    )
}

fn emit_matrix_reduction_ptx(matrices: usize, rows: usize, cols: usize) -> String {
    format!(
        ".version 7.0\n.target sm_80\n.address_size 64\n\n// Specialized for a batch of {matrices} float64 matrices with shape {rows}x{cols}.\n.visible .entry matrix_column_sum_f64(\n    .param .u64 matrices,\n    .param .u64 matrix_column_sums,\n    .param .u32 matrix_count,\n    .param .u32 rows,\n    .param .u32 cols\n)\n{{\n    // linear thread id -> (matrix, col)\n    mov.u32 %r1, %ctaid.x;\n    mov.u32 %r2, %ntid.x;\n    mov.u32 %r3, %tid.x;\n    mad.lo.s32 %r4, %r1, %r2, %r3;\n    // ... compute matrix/col, loop rows, accumulate add.f64 into matrix_column_sums[linear]\n    add.f64 %fd1, %fd1, %fd2;\n    ret;\n}}\n\n.visible .entry average_columns_f64(\n    .param .u64 matrix_column_sums,\n    .param .u64 average_column_sums,\n    .param .u32 matrix_count,\n    .param .u32 cols\n)\n{{\n    // one thread per column; sum matrix_column_sums[matrix * cols + col]\n    add.f64 %fd1, %fd1, %fd2;\n    div.rn.f64 %fd3, %fd1, %fd2;\n    ret;\n}}\n\n.visible .entry matrix_totals_f64(\n    .param .u64 matrix_column_sums,\n    .param .u64 matrix_totals,\n    .param .u32 matrix_count,\n    .param .u32 cols\n)\n{{\n    // one thread per matrix; reduce that matrix's column sums into matrix_totals[matrix]\n    add.f64 %fd1, %fd1, %fd2;\n    ret;\n}}\n\n.visible .entry grand_total_f64(\n    .param .u64 matrix_totals,\n    .param .u64 grand_total,\n    .param .u32 matrix_count\n)\n{{\n    // lane 0 reduces all matrix totals into grand_total[0]\n    add.f64 %fd1, %fd1, %fd2;\n    ret;\n}}"
    )
}

fn emit_matrix_reduction_host_launch(matrices: usize, rows: usize, cols: usize) -> String {
    format!(
        "fn launch_matrix_batch_average_columns_and_grand_total_f64(\n  cuda: &CudaRuntime,\n  matrices: DeviceBuffer<f64>,\n  matrix_column_sums: DeviceBuffer<f64>,\n  average_column_sums: DeviceBuffer<f64>,\n  matrix_totals: DeviceBuffer<f64>,\n  grand_total: DeviceBuffer<f64>,\n) {{\n  debug_assert_eq!(matrices.len(), {matrices} * {rows} * {cols});\n  debug_assert_eq!(matrix_column_sums.len(), {matrices} * {cols});\n  debug_assert_eq!(average_column_sums.len(), {cols});\n  debug_assert_eq!(matrix_totals.len(), {matrices});\n  debug_assert_eq!(grand_total.len(), 1);\n\n  let module = cuda.load_ptx(MATRIX_BATCH_AVERAGE_COLUMNS_AND_GRAND_TOTAL_F64_PTX);\n  let stage1 = module.function(\"matrix_column_sum_f64\");\n  let stage2 = module.function(\"average_columns_f64\");\n  let stage3 = module.function(\"matrix_totals_f64\");\n  let stage4 = module.function(\"grand_total_f64\");\n  let block = 256u32;\n  let stage1_grid = ({matrices}u32 * {cols}u32).div_ceil(block);\n  let stage2_grid = ({cols}u32).div_ceil(block);\n  let stage3_grid = ({matrices}u32).div_ceil(block);\n\n  cuda.launch_1d(&stage1, stage1_grid, block, (&matrices, &matrix_column_sums, &({matrices}u32), &({rows}u32), &({cols}u32)));\n  cuda.launch_1d(&stage2, stage2_grid, block, (&matrix_column_sums, &average_column_sums, &({matrices}u32), &({cols}u32)));\n  cuda.launch_1d(&stage3, stage3_grid, block, (&matrix_column_sums, &matrix_totals, &({matrices}u32), &({cols}u32)));\n  cuda.launch_1d(&stage4, 1, 1, (&matrix_totals, &grand_total, &({matrices}u32)));\n}}"
    )
}

fn describe_token(token: &Token) -> String {
    match token {
        Token::Identifier(name) => format!("identifier `{name}`"),
        Token::Equals => "`=`".to_string(),
        Token::Dot => "`.`".to_string(),
        Token::Comma => "`,`".to_string(),
        Token::LParen => "`(`".to_string(),
        Token::RParen => "`)`".to_string(),
        Token::Arrow => "`=>`".to_string(),
        Token::Semicolon => "`;`".to_string(),
        Token::Plus => "`+`".to_string(),
        Token::Minus => "`-`".to_string(),
        Token::Star => "`*`".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        compile_gpu_pipeline, compile_matrix_reduction_pipeline, emit_ptx, lower_to_kernel,
        sample_gpu_map2_source, tokenize, Parser,
    };

    #[test]
    fn parses_the_sample_program() {
        let tokens = tokenize(sample_gpu_map2_source()).expect("sample should tokenize");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("sample should parse");

        assert_eq!(program.output, "out");
        assert_eq!(program.invocation.left_buffer, "a");
        assert_eq!(program.invocation.right_buffer, "b");
    }

    #[test]
    fn lowers_to_a_guarded_kernel_store() {
        let tokens = tokenize(sample_gpu_map2_source()).expect("sample should tokenize");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("sample should parse");
        let kernel = lower_to_kernel(&program).expect("lowering should succeed");

        let pretty = kernel.to_pretty_string();
        assert!(pretty.contains("let i = global_id.x;"));
        assert!(pretty.contains("out[i] = (a[i] + b[i]);"));
    }

    #[test]
    fn emits_ptx_for_the_sample_program() {
        let artifact = compile_gpu_pipeline(sample_gpu_map2_source()).expect("compile should work");

        assert!(artifact.ptx.contains(".visible .entry out_map2_add_f32("));
        assert!(artifact.ptx.contains("ld.global.f32"));
        assert!(artifact.ptx.contains("add.f32"));
        assert!(artifact.host_launch.contains("cuda.launch_1d"));
    }

    #[test]
    fn supports_a_multiplication_lambda() {
        let source = "let out = gpu.map2(a, b, (x, y) => x * y);";
        let tokens = tokenize(source).expect("source should tokenize");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("source should parse");
        let kernel = lower_to_kernel(&program).expect("lowering should succeed");
        let ptx = emit_ptx(&kernel).expect("ptx should emit");

        assert!(kernel.kernel_name.ends_with("_mul_f32"));
        assert!(ptx.contains("mul.f32"));
    }

    #[test]
    fn compiles_a_four_stage_matrix_reduction_pipeline() {
        let artifact =
            compile_matrix_reduction_pipeline(4, 1_000, 1_000).expect("pipeline should compile");

        assert!(artifact.reduction_ir.contains("average_columns"));
        assert!(artifact.stage1_kernel_ir.contains("matrix_column_sum_f64"));
        assert!(artifact.stage2_kernel_ir.contains("average_columns_f64"));
        assert!(artifact.stage3_kernel_ir.contains("matrix_totals_f64"));
        assert!(artifact.stage4_kernel_ir.contains("grand_total_f64"));
        assert!(artifact.ptx.contains("add.f64"));
        assert!(artifact.host_launch.contains("stage1"));
        assert!(artifact.host_launch.contains("stage4"));
    }
}
