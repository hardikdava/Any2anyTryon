import time
from typing import Optional, List

import numpy as np
from PIL import Image
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
from diffusers import FluxInpaintPipeline, AutoencoderKL
from src.pipeline_tryon import FluxTryonPipeline, crop_to_multiple_of_16, resize_and_pad_to_size, resize_by_height

import torch


MODEL_CACHE = "FLUX.1-dev"
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
MODEL_URL = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)


"""
copied from https://github.com/nftblackmagic/catvton-flux/blob/main/tryon_inference.py
"""

class Any2AnyTryOn:

    def __init__(self):
        """
        Initialize the In-Context Multi-LoRA model.
        """
        super().__init__()

        self.torch_dtype = torch.float16

        text_encoder = CLIPTextModel.from_pretrained(
            MODEL_CACHE, subfolder="text_encoder", torch_dtype=self.torch_dtype
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            MODEL_CACHE, subfolder="text_encoder_2", torch_dtype=self.torch_dtype
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_CACHE, subfolder="transformer", torch_dtype=self.torch_dtype
        )
        vae = AutoencoderKL.from_pretrained(
            MODEL_CACHE, subfolder="vae", torch_dtype=self.torch_dtype
        )

        self.pipe = FluxTryonPipeline.from_pretrained(
            MODEL_CACHE,
            transformer=transformer,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            vae=vae,
            torch_dtype=self.torch_dtype,
        )

        lora_name = "dev_lora_any2any_tryon.safetensors"
        self.pipe.load_lora_weights(
            "loooooong/Any2anyTryon",
            weight_name=lora_name,
            adapter_name="tryon",
        )
        self.pipe.to("cuda")

    def predict(self,
                image_path: str,
                garment_path: str,
                mask_path: str,
                guidance_scale: float = 3.5,
                num_inference_steps: int = 30,
                seed: Optional[int] = 42,
                height: int = 384,
                width: int = 512,
                prompt: str = "",
                num_outputs: int = 1,
                ):
        # Load and process images
        height, width = int(height), int(width)
        width = width - (width % 16)
        height = height - (height % 16)

        concat_image_list = [
            Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
        ]

        model_image = Image.open(image_path)
        print("image_path", image_path)
        input_height, input_width = model_image.size[1], model_image.size[0]
        model_image, lp, tp, rp, bp = resize_and_pad_to_size(
            model_image, width, height
        )
        concat_image_list.append(model_image)

        garment_image = Image.open(garment_path)
        print("garment_image", garment_image)
        garment_image = resize_by_height(garment_image, height)
        concat_image_list.append(garment_image)

        image = Image.fromarray(
            np.concatenate([np.array(img) for img in concat_image_list], axis=1)
        )

        mask_image = Image.open(mask_path)
        print("mask_image", mask_image)

        start_time = time.time()
        print("Using seed:", seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        print("Running inference...")
        results = self.pipe(
            prompt,
            image=image,
            mask_image=mask_image,
            strength=1.0,
            height=height,
            width=image.width,
            target_width=width,
            tryon=True,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=generator,
            output_type="pil",
        )
        outputs = []
        for result in results.images:
            image = result.crop(
                (lp, tp, image.width - rp, image.height - bp)
            ).resize((input_width, input_height))
            outputs.append(image)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        return outputs
