from model3d import Model3d
from consts import IMAGE_PATH, BOX_QUEUE

model = Model3d(imagesPath=IMAGE_PATH, boxQueue=BOX_QUEUE)
model.run()
