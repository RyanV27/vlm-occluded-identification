import json
import argparse

from pathlib import Path
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Prompts the VLM with the environment images to identify the objects in each scene.")

    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        choices=["Qwen2.5-VL", "InternVL3.5", "Molmo2", "FT-Qwen2.5-VL", 
                 "Llama-3.2", "Gemma3", "FT-Llama-3.2", "FT-Gemma3", "Qwen3-VL", 
                 "FT-Qwen3-VL"],
        help="Name of the VLM to test."
    )
    parser.add_argument("--occ_level",  required=True, choices=["no", "low", "medium", "high"])
    parser.add_argument(
        "--out_path",
        type=str,
        default="./runs/",
        help=(
            "Path to output directory where the model outputs should be stored."
        )
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/scratch/rsvargh2/synthetic_occlusion_images/coco/object",
        help=(
            "Path to data directory where environment images and a JSON",
            "file where an object list for each environment is stored."
        )
    )
    parser.add_argument(
        "--model_cache_path",
        type=str,
        default="/scratch/rsvargh2/huggingface_models/",
        help=(
            "Path to data directory where models cache files are stored."
        )
    )

    args = parser.parse_args()

    # Setting path variables
    data_path = Path(args.data_path) / args.occ_level
    metadata_file_path = data_path / "metadata.json"

    # Reading the metadata about the images
    with open(metadata_file_path, "r") as f_in:
        metadata = json.load(f_in)

    # Setting and creating a file to store model generations
    out_dir_path = Path(args.out_path) / args.model_name
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_path / f"{args.model_name}_{args.occ_level}_generations.json"

    # Load existing generations if the output file already exists
    if out_path.exists():
        with open(out_path, "r") as f_in:
            overall_scenes_generations = json.load(f_in)
        print(f"Loaded {len(overall_scenes_generations)} existing generations from {out_path}")
        if len(overall_scenes_generations) == len(metadata):
            print(f"All generations are present for {args.occ_level} occlusion level and {args.model_name} model. Exiting code.")
            return
    else:
        overall_scenes_generations = {}

    # Loading the model to test
    if args.model_name == "Qwen2.5-VL":
        from models import Qwen25VL
        model = Qwen25VL(cache_dir=args.model_cache_path)
    elif args.model_name == "InternVL3.5":
        from models import InternVL35
        model = InternVL35(cache_dir=args.model_cache_path)
    elif args.model_name == "Molmo2":
        from models import Molmo2
        model = Molmo2(cache_dir=args.model_cache_path)
    elif args.model_name == "Molmo":
        from models import Molmo
        model = Molmo(cache_dir=args.model_cache_path)
    elif args.model_name == "FT-Qwen2.5-VL":
        from models import FTQwen25VL
        model = FTQwen25VL()
    elif args.model_name == "Gemma3":
        from models import Gemma3
        model = Gemma3()
    elif args.model_name == "FT-Gemma3":
        from models import FTGemma3
        model = FTGemma3()
    elif args.model_name == "Llama-3.2":
        from models import Llama3_2
        model = Llama3_2()
    elif args.model_name == "FT-Llama-3.2":
        from models import FTLlama3_2
        model = FTLlama3_2()
    elif args.model_name == "Qwen3-VL":
        from models import Qwen3VL
        model = Qwen3VL()
    elif args.model_name == "FT-Qwen3-VL":
        from models import FTQwen3VL
        model = FTQwen3VL()

    # Setting the prompt
    prompt = "List everything that is visible in the image."

    count = 0
    for scene in metadata:
        scene_id = scene['image_id']

        # Skip if this scene has already been processed
        if str(scene_id) in overall_scenes_generations:
            print(f"\nImg {scene_id} already processed, skipping.")
            continue

        print(f"\nImg {scene_id}")

        scene_generations = scene.copy()
    
        image_file_name = scene['file_path'].split('/')[-1]
        image_path = data_path / image_file_name
        print(image_path)
        
        scene_generations['generation'], _, _ = model.run_inference(prompt=prompt, image_path=str(image_path.resolve()))
        print(f"\nGeneration: {scene_generations['generation']}")

        overall_scenes_generations[str(scene_id)] = scene_generations
        
        count += 1
        if count % 10 == 0:
            with open(out_path, "w") as f_out:
                json.dump(overall_scenes_generations, f_out)

    with open(out_path, "w") as f_out:
        json.dump(overall_scenes_generations, f_out)
        

if __name__ == "__main__":
    main()