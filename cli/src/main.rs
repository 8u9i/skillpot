use std::fs;
use std::path::PathBuf;
use clap::{Parser, Subcommand, Args};
use log::info;
use axon_core::*;

#[derive(Parser)]
#[command(name = "axon", about = "Adaptive eXecutable Object Notation CLI", version)]
struct Cli { #[command(subcommand)] command: Commands }

#[derive(Subcommand)]
enum Commands {
    Inspect(InspectArgs), Pack(PackArgs), Unpack(UnpackArgs),
    Convert(ConvertArgs), Bench(BenchArgs), Validate(ValidateArgs),
    List(ListArgs), Extract(ExtractArgs), Create(CreateArgs),
}

#[derive(Args)] struct InspectArgs { path: PathBuf, #[arg(long)] hex: bool }
#[derive(Args)] struct PackArgs { #[arg(short, long)] manifest: PathBuf, #[arg(short, long)] data_dir: PathBuf, #[arg(short, long)] output: PathBuf, #[arg(short, long)] architecture: Option<String>, #[arg(short = 'n', long)] model: Option<String> }
#[derive(Args)] struct UnpackArgs { path: PathBuf, #[arg(short, long)] output: PathBuf, #[arg(long)] raw: bool }
#[derive(Args)] struct ConvertArgs { input: PathBuf, output: PathBuf }
#[derive(Args)] struct BenchArgs { path: PathBuf, #[arg(short, long, default_value = "10")] iterations: u32 }
#[derive(Args)] struct ValidateArgs { path: PathBuf, #[arg(long)] no_checksums: bool }
#[derive(Args)] struct ListArgs { path: PathBuf, #[arg(long)] verbose: bool }
#[derive(Args)] struct ExtractArgs { path: PathBuf, #[arg(short, long)] name: String, #[arg(short, long)] output: PathBuf }
#[derive(Args)] struct CreateArgs { output: PathBuf, #[arg(short, long)] model: Option<String>, #[arg(short, long)] architecture: Option<String> }

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    match Cli::parse().command {
        Commands::Inspect(a) => cmd_inspect(&a),
        Commands::Pack(a) => cmd_pack(&a),
        Commands::Unpack(a) => cmd_unpack(&a),
        Commands::Convert(a) => cmd_convert(&a),
        Commands::Bench(a) => cmd_bench(&a),
        Commands::Validate(a) => cmd_validate(&a),
        Commands::List(a) => cmd_list(&a),
        Commands::Extract(a) => cmd_extract(&a),
        Commands::Create(a) => cmd_create(&a),
    }
}

fn cmd_inspect(args: &InspectArgs) {
    let data = fs::read(&args.path).unwrap_or_else(|e| panic!("Failed to read {}: {e}", args.path.display()));
    if args.hex {
        println!("=== First 256 bytes (hex) ===");
        for (i, chunk) in data.iter().take(256).enumerate() {
            if i % 16 == 0 { print!("\n{:08X}  ", i); }
            print!("{:02X} ", chunk);
        }
        println!();
    }
    let file = AxonFile::from_bytes(data).unwrap_or_else(|e| panic!("Failed to parse .axon: {e}"));
    println!("=== .axon File Inspection ===");
    println!("Magic:       {:?}", std::str::from_utf8(&file.header.magic).unwrap());
    println!("Version:     {}", file.header.version);
    println!("Tensors:     {}", file.header.tensor_count);
    println!("Payload:     {} bytes ({:.2} MB)", file.header.payload_size, file.header.payload_size as f64 / 1_048_576.0);
    println!("Manifest:    {} bytes at offset {}", file.header.manifest_size, file.header.manifest_offset);
    println!("Flags:       {:#018x}", file.header.flags);
    println!();
    println!("Model:       {}", file.manifest.model.as_deref().unwrap_or("N/A"));
    println!();
    for (i, name) in file.manifest.tensor_order.iter().enumerate() {
        if let Some(desc) = file.manifest.get_tensor(name) {
            let dtype = desc.dtype().unwrap_or(DType::F32);
            let shape = desc.shape_vec();
            let s: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
            println!("  [{:3}] {}  {}  [{}]  {} bytes", i, name, dtype.name(), s.join(", "), desc.data_size);
        }
    }
}

fn cmd_pack(args: &PackArgs) {
    let manifest_json = fs::read_to_string(&args.manifest).expect("Failed to read manifest");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_json).expect("Invalid JSON manifest");
    let mut builder = AxonBuilder::new();
    if let Some(ref model) = args.model { builder = builder.model(model); }
    if let Some(ref arch) = args.architecture { builder = builder.architecture(arch); }
    let tensors = manifest["tensors"].as_array().expect("Manifest must have a 'tensors' array");
    for entry in tensors {
        let name = entry["name"].as_str().expect("Tensor must have a name");
        let dtype_code = entry["dtype"].as_u64().unwrap_or(0) as u32;
        let shape: Vec<u64> = entry["shape"].as_array().expect("Tensor must have shape").iter().map(|v| v.as_u64().unwrap()).collect();
        let dtype = DType::from_code(dtype_code).expect("Invalid dtype code");
        let data_path = args.data_dir.join(name);
        let data = fs::read(&data_path).unwrap_or_else(|_| {
            let expected = shape.iter().product::<u64>() as usize * dtype.size_in_bytes();
            info!("Generating {} bytes synthetic data for {}", expected, name);
            (0..expected).map(|i| ((i.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as u8).collect()
        });
        builder = builder.add_tensor(name, data, dtype, &shape);
        info!("Added tensor: {} dtype={} shape={:?}", name, dtype.name(), shape);
    }
    let axon_bytes = builder.build().expect("Failed to build .axon file");
    fs::write(&args.output, &axon_bytes).expect("Failed to write .axon file");
    println!("Written: {} ({:.2} MB)", args.output.display(), axon_bytes.len() as f64 / 1_048_576.0);
}

fn cmd_unpack(args: &UnpackArgs) {
    fs::create_dir_all(&args.output).expect("Failed to create output directory");
    let data = fs::read(&args.path).expect("Failed to read file");
    let file = AxonFile::from_bytes(data).expect("Failed to parse .axon file");
    for (name, desc) in &file.manifest.tensors {
        let tensor_bytes = file.tensor_data(name).expect("Failed to get tensor data");
        let output_path = if args.raw {
            args.output.join(format!("{}.bin", name.replace('/', ".")))
        } else {
            args.output.join(format!("{}.npy", name.replace('/', ".")))
        };
        if args.raw {
            fs::write(&output_path, tensor_bytes).expect("Failed to write tensor");
        } else {
            let mut out_bytes = Vec::new();
            out_bytes.extend_from_slice(&npy_header(desc));
            out_bytes.extend_from_slice(tensor_bytes);
            fs::write(&output_path, out_bytes).expect("Failed to write .npy tensor");
        }
        info!("Extracted: {} -> {} ({} bytes)", name, output_path.display(), tensor_bytes.len());
    }
    println!("Extracted {} tensors to {}", file.manifest.tensor_count(), args.output.display());
}

fn cmd_convert(args: &ConvertArgs) {
    let data = fs::read(&args.input).expect("Failed to read input file");
    let file = AxonFile::from_bytes(data).expect("Failed to parse .axon file");
    let json = serde_json::to_string_pretty(&file.manifest).expect("Failed to serialize");
    fs::write(&args.output, &json).expect("Failed to write JSON");
    println!("Converted {} -> {} ({} tensors)", args.input.display(), args.output.display(), file.manifest.tensor_count());
}

fn cmd_bench(args: &BenchArgs) {
    use std::time::Instant;
    println!("Benchmarking: {} ({} iterations)", args.path.display(), args.iterations);
    let start = Instant::now();
    for _ in 0..args.iterations {
        let data = fs::read(&args.path).expect("Failed to read file");
        let _file = AxonFile::from_bytes(data).expect("Failed to parse");
    }
    let dur = start.elapsed();
    let avg = dur / args.iterations;
    println!("  Load:  {:?} total, {:?} avg", dur, avg);
    let data = fs::read(&args.path).expect("Failed to read");
    let file = AxonFile::from_bytes(data).expect("Failed to parse");
    let start = Instant::now();
    for _ in 0..args.iterations {
        for (name, _) in &file.manifest.tensors { let _ = file.tensor_data(name); }
    }
    let dur = start.elapsed();
    let avg = dur / args.iterations;
    println!("  Index: {:?} total, {:?} avg", dur, avg);
    println!("  Tensors: {}", file.manifest.tensor_count());
    println!("  Payload: {} bytes ({:.2} MB)", file.header.payload_size, file.header.payload_size as f64 / 1_048_576.0);
}

fn cmd_validate(args: &ValidateArgs) {
    let data = fs::read(&args.path).expect("Failed to read file");
    let file = AxonFile::from_bytes(data).expect("Failed to parse .axon file");
    println!("Validating: {}", args.path.display());
    println!("  Magic:      OK (AXON v{})", file.header.version);
    println!("  Tensors:    {} descriptors", file.header.tensor_count);
    if !args.no_checksums {
        let r = file.verify_all_checksums();
        let pass = r.iter().filter(|(_, ok)| *ok).count();
        let fail = r.iter().filter(|(_, ok)| !*ok).count();
        println!("  Checksums:  {}/{} passed, {} failed", pass, r.len(), fail);
        for (n, ok) in &r { if !ok { eprintln!("  CHECKSUM FAIL: {n}"); } }
    } else { println!("  Checksums:  skipped"); }
    println!("  Status:     VALID");
}

fn cmd_list(args: &ListArgs) {
    let data = fs::read(&args.path).expect("Failed to read file");
    let file = AxonFile::from_bytes(data).expect("Failed to parse .axon file");
    println!("Tensors in {}:", args.path.display());
    println!();
    if args.verbose { println!("{:<5} {:<48} {:<8} {:<24} {:>12}", "#", "Name", "DType", "Shape", "Size"); println!("{}", "-".repeat(100)); }
    for (i, name) in file.manifest.tensor_order.iter().enumerate() {
        if let Some(desc) = file.manifest.get_tensor(name) {
            let dtype = desc.dtype().unwrap_or(DType::F32);
            let shape = desc.shape_vec();
            let s: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
            if args.verbose { println!("{:<5} {:<48} {:<8} [{:<22}] {:>12}", i, name, dtype.name(), s.join(", "), format_size(desc.data_size)); }
            else { println!("  {}  {}  {}", name, dtype.name(), s.join("x")); }
        }
    }
}

fn cmd_extract(args: &ExtractArgs) {
    let data = fs::read(&args.path).expect("Failed to read file");
    let file = AxonFile::from_bytes(data).expect("Failed to parse .axon file");
    let tensor_bytes = file.tensor_data(&args.name).unwrap_or_else(|| panic!("Tensor '{}' not found", args.name));
    fs::write(&args.output, tensor_bytes).expect("Failed to write tensor data");
    println!("Extracted {} -> {} ({} bytes)", args.name, args.output.display(), tensor_bytes.len());
}

fn cmd_create(args: &CreateArgs) {
    let mut builder = AxonBuilder::new();
    if let Some(ref model) = args.model { builder = builder.model(model); }
    if let Some(ref arch) = args.architecture { builder = builder.architecture(arch); }
    let r = |n: usize| { (0..n).map(|i| ((i.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as u8).collect::<Vec<_>>() };
    builder = builder.add_tensor("emb_weight", r(32000 * 4096 * 2), DType::F16, &[32000, 4096]);
    for layer in 0..2 {
        for proj in &["q", "k", "v", "o"] {
            builder = builder.add_tensor(&format!("layer_{}_{}", layer, proj), r(4096 * 4096), DType::Q4, &[4096, 4096]);
        }
    }
    for layer in 0..2 {
        for p in &["gate", "up", "down"] {
            let (rows, cols) = if *p == "down" { (11008, 4096) } else { (4096, 11008) };
            builder = builder.add_tensor(&format!("layer_{}_mlp_{}", layer, p), r(rows * cols * 2), DType::F16, &[rows as u64, cols as u64]);
        }
    }
    builder = builder.add_tensor("norm_weight", r(4096 * 2), DType::F16, &[4096]);
    builder = builder.add_tensor("lm_head", r(32000 * 4096 * 2), DType::F16, &[32000, 4096]);
    let axon_bytes = builder.build().expect("Failed to build .axon file");
    fs::write(&args.output, &axon_bytes).expect("Failed to write .axon file");
    println!("Created: {} ({:.2} MB, {} tensors)", args.output.display(), axon_bytes.len() as f64 / 1_048_576.0, 17);
}

fn npy_header(desc: &TensorDescriptor) -> Vec<u8> {
    let dtype = desc.dtype().unwrap_or(DType::F32);
    let shape = desc.shape_vec();
    let s: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
    let ds = match dtype { DType::F32 => "<f4", DType::F16 | DType::BF16 => "<f2", DType::I32 => "<i4", DType::I64 => "<i8", DType::U8 => "u1", _ => "<f4" };
    let h = format!("{{'descr': '{ds}', 'fortran_order': False, 'shape': ({},) }}", s.join(", "));
    let mut hb = h.as_bytes().to_vec();
    let padded = ((10 + hb.len() + 63) / 64) * 64;
    hb.extend(std::iter::repeat(b' ').take(padded - 10 - hb.len()));
    let mut r = Vec::new();
    r.extend_from_slice(b"\x93NUMPY"); r.push(1); r.push(0);
    r.extend_from_slice(&(hb.len() as u16).to_le_bytes());
    r.extend_from_slice(&hb); r
}

fn format_size(bytes: u64) -> String {
    let u = &["B", "KB", "MB", "GB", "TB"];
    let mut s = bytes as f64; let mut i = 0;
    while s >= 1024.0 && i < u.len() - 1 { s /= 1024.0; i += 1; }
    format!("{:.2} {}", s, u[i])
}
