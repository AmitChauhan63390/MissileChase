from ursina import *
from ursina.camera import Camera  # Add this import
import math
import random
import numpy as np
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.shaders import lit_with_shadows_shader

app = Ursina(size=(1580, 820))

# Ground and environment
ground = Entity(model='plane', texture='white_cube', scale=(200, 1, 200), 
                texture_scale=(200, 200), color=color.green, collider='box')
Sky()

# Use an alternative camera controller
editor_camera = EditorCamera()
editor_camera.position = (0, 10, -30)  # Set initial position

# Text and HUD
Text.default_resolution = 1080 * Text.size
instruction_text = Text(text="Click on ground to move target | Press SPACE to launch missile", 
                       origin=(0, -6), scale=2)
status_text = Text(text="Missiles Left: 1000 | Score: 0", origin=(0, -7), scale=1.5)

# Radar display


missile_cam_panel = Panel(
    parent=camera.ui,
    scale=(0.3, 0.25),  # Adjust size as needed
    position=(0.65, -0.35),  # Bottom right corner
    color=color.black66,
    enabled=False
)


missile_cam_label = Text(
    parent=missile_cam_panel,
    text="MISSILE CAM",
    origin=(0, 0.4),
    color=color.red,
    scale=1.5
)

missile_camera = Entity(
    model='cube',
    color=color.clear,
    scale=(0.1, 0.1, 0.1)
)

missile_cam = Camera()  # Create camera without parameters first
missile_cam.parent = missile_camera  # Then set parent separately
missile_cam.fov = 144
missile_cam.enabled = False
missile_cam.ui = missile_cam_panel


radar_panel = Panel(
    parent=camera.ui,
    scale=(0.2, 0.2),
    position=(0.8, 0.4),
    color=color.black
)

# Center dot (representing the target)
radar_center = Entity(
    parent=radar_panel, 
    model='circle', 
    color=color.red, 
    scale=0.05, 
    position=(0, 0, -0.01)
)

# Missile blip on radar
missile_blip = Entity(
    parent=radar_panel,
    model='circle',
    color=color.cyan,
    scale=0.03,
    position=(0, 0, -0.01),
    enabled=False
)

# Missile distance text
missile_info_text = Text(
    parent=camera.ui,
    text="",
    position=(0.8, 0.5),
    origin=(0, 0),
    scale=1.5,
    color=color.yellow
)

# Global variables
defense_missiles = 1000
score = 0
missile_launcher_pos = Vec3(0, 0.5, 0)
launcher = None
defense_missile = None
target_pos = Vec3(50, 5, 50)  # Initial target position
target = None
missile_launched = False
missile_start_time = 0  # For spiral timing
gravity = Vec3(0, -9.8, 0)
wind_force = Vec3(random.uniform(-1, 1), 0, random.uniform(-1, 1))

# Simplified Kalman filter for 3D position tracking
class SimpleKalmanFilter3D:
    def __init__(self, process_noise=1.0, measurement_noise=25.0):
        # State: [x, vx, y, vy, z, vz]   
        self.x = np.zeros(6)
        self.P = np.eye(6) * 50  # Covariance
        self.R = measurement_noise  # Measurement noise
        self.Q = process_noise  # Process noise
        
    def predict(self, dt=1.0):
        # Simple constant velocity model
        # Update position with velocity
        self.x[0] += self.x[1] * dt
        self.x[2] += self.x[3] * dt
        self.x[4] += self.x[5] * dt
        # Increase uncertainty
        self.P += self.Q
        return self.get_position()
        
    def update(self, measurement):
        # Measurement is [x, y, z]
        # For each dimension, apply Kalman update
        for i in range(3):
            idx = i * 2  # Position index (0, 2, 4)
            pred = self.x[idx]
            K = self.P[idx, idx] / (self.P[idx, idx] + self.R)
            self.x[idx] = pred + K * (measurement[i] - pred)
            self.P[idx, idx] = (1 - K) * self.P[idx, idx]
            # Also update velocity estimate based on change in position
            if K > 0:
                self.x[idx + 1] = (measurement[i] - pred) * 0.1  # Simple velocity estimate
        return self.get_position()
        
    def get_position(self):
        return Vec3(self.x[0], self.x[2], self.x[4])
        
    def get_velocity(self):
        return Vec3(self.x[1], self.x[3], self.x[5])

# Initialize Kalman filter and target
kalman_filter = SimpleKalmanFilter3D()
kalman_estimate = Entity(model='sphere', color=color.white, scale=0.5)

# Create target entity
def create_target():
    global target
    if target:
        destroy(target)
    target = Entity(model='sphere', color=color.red, scale=1, position=target_pos)
    
# Create launcher entity
def place_launcher():
    global launcher
    if launcher:
        destroy(launcher)
    launcher = Entity(model='cube', color=color.gray, scale=(4, 2, 4), 
                     position=missile_launcher_pos, collider='box')

# Custom trail implementation for missile
class MissileTrail:
    def __init__(self, missile, max_points=50, thickness=0.2, color=color.orange):
        self.missile = missile  # Reference to the missile object, not just entity
        self.max_points = max_points
        self.thickness = thickness
        self.color = color
        self.points = []
        self.trail_entities = []
        self.update_rate = 0.05  # How often to add a new point (in seconds)
        self.time_since_last = 0
        
    def update(self, dt):
        self.time_since_last += dt
        # Check if it's time to add a point and the missile is still active
        if self.time_since_last >= self.update_rate and self.missile.active:
            self.time_since_last = 0
            self.add_point(self.missile.entity.position)
            
    def add_point(self, position):
        # Add new point at current missile position
        self.points.append(position)
        
        # Create visual entity for the trail point
        trail_point = Entity(
            model='sphere',
            color=self.color,
            position=position,
            scale=(self.thickness, self.thickness, self.thickness)
        )
        self.trail_entities.append(trail_point)
        
        # Remove oldest points if we exceed max_points
        if len(self.points) > self.max_points:
            old_point = self.points.pop(0)
            destroy(self.trail_entities.pop(0))
            
    def destroy(self):
        # Clean up all trail entities
        for entity in self.trail_entities:
            destroy(entity)
        self.points = []
        self.trail_entities = []

# Missile class with spiral trajectory
class SpiralGuidedMissile:
    def __init__(self, position, target_position, speed=60):
        # Create a simple missile model
        self.entity = Entity(model='sphere', color=color.blue, scale=0.5, position=position)
        
        # Add some visual details to make it look more like a missile
        self.body = Entity(parent=self.entity, model='cube', color=color.light_gray, 
                         scale=(0.2, 1.0, 0.2), position=(0, 0, 0))
        
        # Add fins
        for i in range(4):
            angle = i * 90
            rad_angle = math.radians(angle)
            x = math.sin(rad_angle) * 0.15
            z = math.cos(rad_angle) * 0.15
            fin = Entity(parent=self.entity, model='cube', color=color.dark_gray,
                        scale=(0.05, 0.2, 0.2), position=(x, -0.3, z))
            fin.rotation_y = angle
        
        # CAMERA SETUP - Now that self.entity exists, we can attach the camera mount
        self.camera_mount = Entity(
            parent=self.entity, 
            position=(0, 0, 0.5)  # Position slightly in front of missile
        )
        
        # Attach missile camera to the missile
        missile_camera.parent = self.camera_mount
        missile_camera.position = (0, 0, 0)
        missile_camera.rotation = (0, 0, 0)
        
        # Enable camera and viewport
        missile_cam.enabled = True
        missile_cam_panel.enabled = True
        
        # Rest of your existing initialization code
        self.velocity = Vec3(0, 0, 0)
        self.speed = speed
        self.start_time = time.time()
        self.active = True
        
        # Create trail after setting active status
        self.trail = MissileTrail(missile=self, thickness=0.2, color=color.orange)
        
        # Add thrust particles
        self.particles = []
        self.emit_interval = 0.01
        self.last_emit_time = 0
        
        # Enable radar tracking
        missile_blip.enabled = True
        
    def emit_thrust_particle(self):
        # Create a new thrust particle at the back of the missile
        pos = self.entity.position - self.entity.forward * 0.5
        pos += Vec3(random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))
        
        # Random color between orange and yellow
        r = random.uniform(0.8, 1.0)
        g = random.uniform(0.3, 0.7)
        particle_color = color.rgb(r, g, 0.1)
        
        particle = Entity(
            model='sphere',
            color=particle_color,
            position=pos,
            scale=0.15 + random.uniform(0, 0.1)
        )
        
        # Add to particles list
        self.particles.append((particle, time.time()))
        
        # Limit number of particles
        if len(self.particles) > 40:
            destroy(self.particles[0][0])
            self.particles.pop(0)
            
    def update_particles(self, dt):
        # Update existing particles
        current_time = time.time()
        for i, (particle, spawn_time) in enumerate(self.particles[:]):
            age = current_time - spawn_time
            if age > 0.5:  # Particle lifetime
                destroy(particle)
                self.particles.remove((particle, spawn_time))
            else:
                # Shrink and fade particle
                scale_factor = 1.0 - (age / 0.5)
                particle.scale = 0.15 * scale_factor
                
                # Move particle backward slightly
                particle.position -= self.entity.forward * dt * 2
        
        # Emit new particles
        self.last_emit_time += dt
        if self.last_emit_time >= self.emit_interval:
            self.last_emit_time = 0
            self.emit_thrust_particle()
        
    def update(self, target_position, dt):
        if not self.active:
            return False
            
        # Update trail and particles
        self.trail.update(dt)
        self.update_particles(dt)
            
        # Direction to target
        direction = (target_position - self.entity.position).normalized()
        
        # Spiral effect based on time
        spiral_time = time.time() - self.start_time
        spiral_intensity = min(spiral_time * 0.5, 3.0)  # Increase spiral over time, max out at 3
        
        # Create spiral by adding perpendicular components
        up_vector = Vec3(0, 1, 0)
        right_vector = direction.cross(up_vector).normalized()
        up_vector = right_vector.cross(direction).normalized()
        
        # Compute spiral offset
        spiral_x = math.sin(spiral_time * 3) * spiral_intensity
        spiral_y = math.cos(spiral_time * 3) * spiral_intensity
        
        # Apply spiral to direction
        spiral_direction = direction + (right_vector * spiral_x + up_vector * spiral_y) * 0.1
        spiral_direction = spiral_direction.normalized()
        
        # Set velocity with spiral
        self.velocity = spiral_direction * self.speed
        
        # Apply environmental effects
        self.velocity += gravity * dt
        self.velocity += wind_force * dt
        
        # Update position
        self.entity.position += self.velocity * dt
        
        # Look in the direction of movement
        self.entity.look_at(self.entity.position + self.velocity)
        
        # Update radar display
        relative_pos = self.entity.position - target_position
        radar_x = relative_pos.x / 100  # Scale down to fit radar
        radar_z = relative_pos.z / 100
        missile_blip.position = (radar_x, radar_z, -0.01)
        
        # Update distance text
        dist = distance(self.entity.position, target_position)
        missile_info_text.text = f"Distance: {dist:.1f}m\nSpeed: {self.speed:.1f}m/s"
        
        # Check for proximity to target (hit detection)
        if distance(self.entity.position, target_position) < 2.0:
            print("Hit target!")
            return True
            
        # Check if missile is out of bounds or hits ground
        if self.entity.position.y <= 0.1 or any(abs(coord) > 150 for coord in self.entity.position):
            print("Missile missed or out of bounds")
            return False
            
        return None  # Still active
        
    def destroy(self):
        self.active = False  # Set inactive first
        self.trail.destroy()  # Then destroy trail
        
        # Disable missile camera view
        missile_cam.enabled = False
        missile_cam_panel.enabled = False
        missile_camera.parent = scene  # Reset parent to scene
        
        # Clean up particles
        for particle, _ in self.particles:
            destroy(particle)
        self.particles = []
        
        # Hide radar tracking
        missile_blip.enabled = False
        missile_info_text.text = ""
        
        # Destroy missile entity
        destroy(self.entity)
        

# Launch missile function
def launch_missile():
    global defense_missile, missile_launched, defense_missiles
    if missile_launched or defense_missiles <= 0:
        return
        
    defense_missiles -= 1
    status_text.text = f"Missiles Left: {defense_missiles} | Score: {score}"
    
    # Get Kalman-filtered position for targeting
    estimated_pos = kalman_filter.get_position()
    
    # Launch missile from launcher position
    defense_missile = SpiralGuidedMissile(position=missile_launcher_pos, 
                                         target_position=estimated_pos)
    
    # Set initial velocity toward target
    initial_dir = (estimated_pos - missile_launcher_pos).normalized()
    defense_missile.velocity = initial_dir * defense_missile.speed
    
    missile_launched = True
    print("Missile launched with spiral trajectory!")

# Move target based on mouse click
def move_target():
    global target_pos
    if mouse.hovered_entity == ground:
        # Get the point on the ground where mouse clicked
        ground_point = mouse.world_point
        # Set random height for target
        target_pos = Vec3(ground_point.x, random.uniform(5, 15), ground_point.z)
        
        if target:
            # Add noise to simulate evasive maneuvers
            target.animate_position(
                target_pos + Vec3(random.uniform(-3, 3), random.uniform(-1, 1), random.uniform(-3, 3)), 
                duration=1.0,
                curve=curve.linear
            )

# Update function
def update():
    global missile_launched, score, defense_missile
    
    # Move target with some random movement
    if target:
        # Add small random movement to target
        noise_x = random.uniform(-0.2, 0.2)
        noise_y = random.uniform(-0.1, 0.1)
        noise_z = random.uniform(-0.2, 0.2)
        target.position += Vec3(noise_x, noise_y, noise_z)
        
        # Keep target above ground
        if target.position.y < 3:
            target.position.y = 3
            
        # Add measurement noise (simulate radar noise)
        noisy_pos = target.position + Vec3(
            random.gauss(0, 3),
            random.gauss(0, 1),
            random.gauss(0, 3)
        )
        
        # Update Kalman filter with noisy measurement
        kalman_filter.update([noisy_pos.x, noisy_pos.y, noisy_pos.z])
        
        # Predict next position
        kalman_est_pos = kalman_filter.predict(time.dt)
        
        # Update Kalman estimate visualization
        kalman_estimate.position = kalman_est_pos
    
    # Update missile if launched
    if missile_launched and defense_missile:
        result = defense_missile.update(kalman_estimate.position, time.dt)
        
        if result is True:  # Hit
            defense_missile.destroy()
            missile_launched = False
            score += 10
            status_text.text = f"Missiles Left: {defense_missiles} | Score: {score}"
            
        elif result is False:  # Miss
            defense_missile.destroy()
            missile_launched = False

# Input handling
def input(key):
    if key == 'left mouse down':
        move_target()
    if key == 'space':
        launch_missile()

# Initialize the scene
place_launcher()
create_target()

# Run the app
app.run()