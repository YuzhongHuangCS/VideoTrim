from PIL import Image, ImageFilter
import numpy as np

jpg = np.asarray(Image.open('0_v1.jpg')).astype(float)
png = np.asarray(Image.open('0_v1.png')).astype(float)


diff = Image.fromarray(np.clip(np.abs(jpg-png), 0, 255).astype(np.uint8))
diff.save('diff.png')
