import os
import time
import subprocess
from typing import List

from cog import BasePredictor, Input, Path
import torch
from PIL import Image

from core import Any2AnyTryOn, MODEL_CACHE, MODEL_URL


def download_weights(url, dest, file=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class CogPredictor(BasePredictor):

    def setup(self) -> None:
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, ".")
        self.predictor = Any2AnyTryOn()

    @torch.inference_mode()
    def predict(
            self,
            image: Path = Input(
                description="Input image",
                default=None
            ),
            garment: Path = Input(
                description="Input garment image",
                default=None
            ),
            mask: Path = Input(
                description="Input mask image",
                default=None
            ),
            prompt: str = Input(
                description="Prompt",
                default="",
            ),
            guidance_scale: float = Input(
                description="Guidance scale",
                default=3.5,
                ge=0,
                le=50
            ),
            seed: int = Input(
                description="Random seed. Set for reproducible generation",
                default=42
            ),
            num_inference_steps: int = Input(
                description="Number of inference steps",
                ge=1, le=50, default=30,
            ),
            output_format: str = Input(
                description="Format of the output images",
                choices=["webp", "jpg", "png"],
                default="webp",
            ),
            output_quality: int = Input(
                description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
                default=80,
                ge=0,
                le=100,
            ),
    ) -> list[Path]:
        """Run a single prediction on the model"""
        print("seed:", seed)
        print("guidance scale:", guidance_scale)
        print("num inference steps:", num_inference_steps)

        outputs = self.predictor.predict(
            image_path=str(image),
            garment_path=str(garment),
            mask_path=str(mask),
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        return self.post_process(outputs, output_format, output_quality)

    def post_process(
            self, images: List[Image.Image], output_format="webp", output_quality=80
    ):
        output_paths = []
        for i, image in enumerate(images):
            # TODOs: Add safety checker here
            output_path = f"/tmp/out-{i}.{output_format}"
            if output_format != "png":
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))
        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt."
            )
        return output_paths
