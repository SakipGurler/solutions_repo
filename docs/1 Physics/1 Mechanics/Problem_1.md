# Problem 1

## Investigating the Range as a Function of the Angle of Projection

### 1. Theoretical Foundation

#### Equations of Motion:

- **Horizontal Motion:**
  $ x = v_0 \cos\theta \cdot t $
- **Vertical Motion:**
  $ y = v_0 \sin\theta \cdot t - \frac{1}{2} g t^2 $

#### Time of Flight:

$ T = \frac{2 v_0 \sin\theta}{g} $

#### Range Formula:

$ R = \frac{v_0^2 \sin 2\theta}{g} $

#### Key Insights:

- Maximum range occurs at $ \theta = 45^\circ $.
- Range is symmetric around $ 45^\circ $.
- Higher initial velocity increases range quadratically.

---

### 2. Analysis of the Range

- **Effect of Initial Velocity ($ v_0 $)**: Increasing $ v_0 $ quadratically increases range.
- **Effect of Gravity ($ g $)**: Higher $ g $ reduces range.

---

### 1. Theoretical Foundation

#### Equations of Motion:

- **Horizontal Motion:**
  $ x = v_0 \cos\theta \cdot t $
- **Vertical Motion:**
  $ y = v_0 \sin\theta \cdot t - \frac{1}{2} g t^2 $

#### Time of Flight:

$ T = \frac{2 v_0 \sin\theta}{g} $

#### Range Formula:

$ R = \frac{v_0^2 \sin 2\theta}{g} $

#### Key Insights:

- Maximum range occurs at $ \theta = 45^\circ $.
- Range is symmetric around $ 45^\circ $.
- Higher initial velocity increases range quadratically.

---

### 2. Analysis of the Range

- **Effect of Initial Velocity ($ v_0 $)**: Increasing $ v_0 $ quadratically increases range.
- **Effect of Gravity ($ g $)**: Higher $ g $ reduces range.
- **Effect of Launch Height ($ h $)**: Changes the range formula, requiring numerical solutions.

### 3. Practical Applications

- **Sports**: Optimizing throw angles in basketball or soccer.
- **Engineering**: Calculating trajectories in ballistics.
- **Astrophysics**: Predicting satellite launches considering different gravities.

---

### 4. Implementation (Python Simulation)

This simulation calculates and visualizes the projectile range as a function of the angle of projection.

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # Gravity (m/s^2)
v0 = 20   # Initial velocity (m/s)
angles = np.linspace(0, 90, 100)  # Angle range (degrees)
ranges = (v0**2 * np.sin(2 * np.radians(angles))) / g

# Plot
plt.figure(figsize=(8,5))
plt.plot(angles, ranges, label="Range vs. Angle")
plt.xlabel("Angle (degrees)")
plt.ylabel("Range (m)")
plt.title("Projectile Range as a Function of Angle")
plt.legend()
plt.grid()
plt.show()
```

![alt text](image.png)

---

### 5. Limitations & Extensions

- **No air resistance**: Real-world projectiles experience **drag**, reducing range.
- **Uneven terrain**: Requires numerical integration.
- **Wind effects**: Adds horizontal acceleration component.

#### Future Work:

🔹 Implement numerical solvers to simulate **drag** and **varying launch heights**.
