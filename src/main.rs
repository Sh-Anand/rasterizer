use std::{env, path::Path, process::ExitCode};

use rasterizer::asset::GltfAsset;

const USAGE: &str = "Usage: rasterizer <scene.gltf|scene.glb>\n\nLoad a local glTF asset and print its contents. No rendering yet.";

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let Some(path) = args.next() else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };

    if args.next().is_some() {
        eprintln!("Expected exactly one asset path.\n\n{USAGE}");
        return ExitCode::FAILURE;
    }
    if path == "--help" || path == "-h" {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let path = Path::new(&path);
    let asset = match GltfAsset::load(path) {
        Ok(asset) => asset,
        Err(error) => {
            eprintln!("Failed to load '{}': {error}", path.display());
            return ExitCode::FAILURE;
        }
    };

    let document = &asset.document;
    let primitives: usize = document.meshes().map(|mesh| mesh.primitives().len()).sum();
    println!("Loaded {}", path.display());
    println!("Scenes: {}", document.scenes().len());
    if let Some(scene) = document.default_scene() {
        println!(
            "Default scene: {} ({})",
            scene.index(),
            scene.name().unwrap_or("unnamed")
        );
    } else {
        println!("Default scene: none");
    }
    println!("Nodes: {}", document.nodes().len());
    println!(
        "Meshes: {} ({primitives} primitives)",
        document.meshes().len()
    );
    println!("Materials: {}", document.materials().len());
    println!("Textures: {}", document.textures().len());
    println!("Images decoded: {}", asset.images.len());
    println!("Buffers loaded: {}", asset.buffers.len());
    println!("Cameras: {}", document.cameras().len());
    println!("Animations: {}", document.animations().len());

    let extensions: Vec<_> = document.extensions_used().collect();
    if !extensions.is_empty() {
        println!("Declared extensions: {}", extensions.join(", "));
        println!("Extension metadata is retained; extension behavior is not implemented.");
    }
    ExitCode::SUCCESS
}
