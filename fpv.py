from ursina import *

app = Ursina()

missile = Entity(model='sphere', color=color.red, scale=1)
missile.position = (0,0,0)

# Create a small 2D icon for missile FPV in corner
fpv_icon = Entity(
    parent=camera.ui,
    model='circle',
    color=color.red,
    scale=0.1,
    position=window.top_left + Vec2(0.15, -0.15)
)

def update():
    missile.x += time.dt * 2  # move missile

    # sync FPV icon position with missile x, y in a limited way for simulation
    fpv_icon.position = window.top_left + Vec2(0.15 + missile.x * 0.01, -0.15)

app.run()
