import mujoco
from judo import MODEL_PATH

model_path = MODEL_PATH / "xml/spot_components/spot_trafffic_cone.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

print(model)
print(data)

import matplotlib.pyplot as plt
# visualize the model
renderer = mujoco.Renderer(model)
renderer.update_scene(data)
rgb = renderer.render()
plt.imshow(rgb)
plt.show()
