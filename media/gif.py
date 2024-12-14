from PIL import Image
from pathlib import Path
import glob

if __name__ == "__main__":
    output_frames = Path(__file__).parent.parent / "output_frames"
    assert output_frames.is_dir()

    # Get list of frames
    frames = []
    imgs = glob.glob(f"{output_frames}/frame_*.png")
    imgs.sort()
    
    # Load all frames
    for filename in imgs:
        frames.append(Image.open(filename))
    
    # Save as GIF
    frames[0].save(
        'header.gif',
        save_all=True,
        append_images=frames[1:],
        duration=33.3,  # 30fps = 33.3ms per frame
        loop=0
    )
