from __future__ import division, print_function
import numpy as np
import copy


def no_force(pos, time):
    """
    We can apply some time/position-dependent force to each particle.
    This is the default function, where there is no force applied on the particle.

    Parameters
    ----------
    pos : numpy array
        Position at which the force is felt
    time: float
        Time at which force is felt
    """
    return np.array([0., 0., 0.])


def element_mult(vec_1, vec_2):
    """
    Multiplies each element of vector vec_1 with corresponding element of vec_2

    Parameters
    ----------
    vec_1: numpy array
        First vector for which elements are multiplied
    vec_2: numpy array
        Second vector for which elements are multiplied
    """
    vec = []
    vec.append(vec_1[0] * vec_2[0])
    vec.append(vec_1[1] * vec_2[1])
    vec.append(vec_1[2] * vec_2[2])
    return np.array(vec)


def vector_from(arr=np.array([0, 0, 0])):
    """
    Creates a vpython vector from a numpy array

    Parameters
    ----------
    arr: numpy array
        Array that is converted into a vpythonv ector
    """
    import vpython
    return vpython.vector(arr[0], arr[1], arr[2])


def normalized(arr):
    """
    Returns normalised numpy array.

    Parameters
    ----------
    arr: numpy array
        Array to be normalised
    """
    return arr / np.linalg.norm(arr)


class Particle(object):
    """
    Class representing a particle, which can be attached to a spring.
    Not tied to any visualisation method.
    """
    def __init__(self, pos=np.array([0, 0, 0]), v=np.array([0, 0, 0]), radius=1., 
        inv_mass=0., color=[1, 0, 0], alpha=1., fixed=False, applied_force=no_force,
        q = 0.):
        """
        Parameters
        ----------
        pos: numpy array
            Initial position of particle
        v: numpy array
            Initial velocity of particle
        radius: float
            Radius of particle
        inv_mass: float
            Inverse mass of particle
        color: array
            Color of particle, given in form [R G B]
        alpha: float
            Alpha of particle, 1 is completely opaque, 0 is completely transparent, used in visualisation
        fixed: boolean
            Whether particle can move or not
        applied_force: function taking arguments of: pos(numpy array) and time(float). Returns a numpy array
            Gives the applied force when the particle is at a certain location and time. By default, no force applied on particle.
        q: Float
            Charge on particle
        """
        self.pos = pos
        self.v = v
        self.inv_mass = inv_mass
        self.color = color
        self.fixed = fixed
        self.radius = radius
        self.applied_force = applied_force
        self.total_force = np.array([0, 0, 0])
        self.alpha = alpha
        self.max_point = np.array([None])
        self.min_point = np.array([None])
        self._visualized = False  # Used in visualisation
        self._pointer_assigned = False  # Used when looking at forces using pointers
        self.q = q
        self.initial_v = copy.deepcopy(self.v)
        self.initial_pos = copy.deepcopy(self.pos)
        self.prev_pos = copy.deepcopy(self.pos)
        self.prev_v = copy.deepcopy(self.v)

    def update(self, dt):
        """
        Updates the position of the particle.
        Parameters
        ----------
        dt: float
            Time step to take
        """
        self.v += (dt * self.total_force) * self.inv_mass
        self.pos += (self.v * dt) + ((0.5 * self.total_force * self.inv_mass) * (dt**2))

    @property
    def amplitude(self):
        """
        Property which gives the amplitude of oscillation. Depends on external thing to set max_point and min_point.
        """
        if self.max_point.any() and self.min_point.any():
            _amplitude = np.linalg.norm(self.max_point - self.min_point)
            return _amplitude
        return None


class Spring(object):
    """
    Class representing a spring. Not tied to any visualisation method.
    """

    @property
    def axis(self):
        """
        Property giving the axis, i.e. the vector showing the orientation and length of the spring.
        """
        return (self.particle_2.pos - self.particle_1.pos)

    @property
    def pos(self):
        """
        Property giving the position of one of the ends of the spring.
        """
        return self.particle_1.pos

    def __init__(self, particle_1, particle_2, k, l0=None, radius=0.5, color=[1, 1, 1], alpha=1.):
        """
        Parameters
        ----------
        particle_1: Particle object
            Particle on one end of spring.
        particle_2: Particle object
            Particle on other end of spring.
        radius: float
            Radius of spring.
        color: array
            Color of particle, given in form [R G B]
        alpha: float
            Alpha of particle, 1 is completely opaque, 0 is completely transparent, used in visualisation
        """
        self.particle_1 = particle_1
        self.particle_2 = particle_2
        self.k = k
        self.alpha = alpha
        self.color = color
        self._visualized = False
        self.radius = radius
        if l0:
            self.l0 = l0
        else:
            self.l0 = np.linalg.norm(self.particle_1.pos - self.particle_2.pos)

    def force_on(self, particle, if_at=np.array([None])):
        """
        Given an arbitary particle, gives the force on that particle. No force if the spring isn't connected to that particle.
        Parameters
        ----------
        particle: Particle object
            Particle which feels the force
        if_at: NumPy Array
            If this parameter is used, this gives the force felt if the particle were at this position.
        """
        if not if_at.any():
            if_at = particle.pos
        if particle == self.particle_1:
            x = np.linalg.norm(if_at - self.particle_2.pos) - self.l0
            return normalized(self.particle_2.pos - if_at) * self.k * x
        elif particle == self.particle_2:
            x = np.linalg.norm(if_at - self.particle_1.pos) - self.l0
            return normalized(self.particle_1.pos - if_at) * self.k * x
        return np.array([0, 0, 0])


class Pointer(object):
    """
    Class representing an arrow/pointer, that isn't tied to any specific visualization method/library.
    """
    def __init__(self, pos, axis, shaftwidth=1, color=[1, 1, 1], alpha=1):
        """
        Parameters
        ----------
        pos: numpy array
            Location of tail end of pointer
        axis: numpy array
            A vector giving the length and orientation of the pointer
        shaftwidth: float
            The width of the pointer's shaft
        color: array
            Color of particle, given in form [R G B]
        alpha: float
            Alpha of particle, 1 is completely opaque, 0 is completely transparent, used in visualisation
        """
        self.pos = pos
        self.axis = axis
        self.shaftwidth = shaftwidth
        self.alpha = alpha
        self.color = color
        self._visualized = False


class System(object):
    """
    Class representing a collection of particles, springs, pointers, and a container(not yet implemented).
    Also can handle visualisation.
    If want to use anything other than vpython, subclass and change create_vis, update_vis, (necessary),
    and maybe run_for (If want a convenience function, but can just use simulate).
    Alternatively, could just put visualize = True and extract information from particles, etc. and visualize.
    """

    def __init__(self,visualize=True, visualizer_type="vpython", particles=[], springs=[],
        canvas=None, stop_on_cycle=False, record_amplitudes=False, display_forces=False):
        """
        Parameters
        ----------
        visualize: boolean
            Whether system visualizes itself.
        visualizer_type: string
            What type of visualizer is used. Used to make sure that some vpython-specific non-essential things don't run.
            Set to anything you like if using another visualization method.
        particles: array of particles
            An array of all the particles in this system.
        springs: array of springs
            An array of all the springs in this system.
        canvas: some view
            Some view to draw everything into. By default, a vpython canvas.
        stop_on_cycle: boolean
            Whether the system stops running when a full cycle is done. Only tested on relatively simple 1-D systems, but seems to work.
        record_amplitudes: boolean
            Whether the system records the amplitudes of oscillations. Only works on simple 1-D systems going along the x-axis.
        display_forces: boolean
            Whether the forces applied onto a particle are displayed.
            By default, these vectors are displaced from the particles by 2*particle radius in y-direction.
            Look in simulate method to change this.
        """
        self.visualize = visualize
        self.particles = particles
        self.springs = springs
        self.pointers = []  # Only used for displaying forces
        # visual representations of the more abstract classes particles, springs, and pointers.
        # By default, vpython is used for these.
        self.spheres = []
        self.helices = []
        self.arrows = []

        self.scene = None
        self.stop_on_cycle = stop_on_cycle
        self.record_amplitudes = record_amplitudes
        self.time = 0.
        self.display_forces = display_forces
        if display_forces:
            self._assign_forces()
        if visualize:
            self.visualizer_type = visualizer_type
            self.create_vis(canvas=canvas)
        else:
            self.visualizer_type = None

    def create_vis(self, canvas=None):
        """
        Creates a visualisation. By default, uses a vpython canvas, so imports vpython.
        Subclass class and change this to change the way visualisations are made.
        Parameters
        ----------
        canvas: vpython canvas
            display into which the visualization is drawn
        """
        import vpython
        #self.scene stores the display into which the system draws
        if not canvas:
            if not self.scene:
                self.scene = vpython.canvas()
        if canvas:
            self.scene = canvas

        # Draw particles if they aren't drawn yet
        for particle in self.particles:
            if not particle._visualized:
                self.spheres.append(vpython.sphere(pos=vector_from(particle.pos),
                    radius=particle.radius,
                    color=vector_from(particle.color),
                    opacity=particle.alpha,
                    display=self.scene))
                particle._visualized = True
        # Draw springs if they aren't drawn yet
        for spring in self.springs:
            if not spring._visualized:
                self.helices.append(vpython.helix(pos=vector_from(spring.particle_1.pos),
                    axis=vector_from(spring.axis),
                    radius=spring.radius,
                    opacity=spring.alpha,
                    color=vector_from(spring.color),
                    display=self.scene))
                spring._visualized = True
        # Draw pointers if they aren't drawn yet
        for pointer in self.pointers:
            if not pointer._visualized:
                self.arrows.append(vpython.arrow(pos=vector_from(pointer.pos),
                    axis=vector_from(pointer.axis),
                    shaftwidth=pointer.shaftwidth,
                    opacity=pointer.alpha,
                    color=vector_from(pointer.color),
                    display=self.scene))
                pointer._visualized = True

    def update_vis(self):
        """
        Function which updates the visualization. In this case, updates the vpython visualization
        """
        import vpython
        # Update display of particles(rendered as spheres)
        for (index, __) in enumerate(self.spheres):
            self.spheres[index].pos = vector_from(self.particles[index].pos)
            self.spheres[index].radius = self.particles[index].radius
            self.spheres[index].color = vector_from(self.particles[index].color)
            self.spheres[index].opacity = self.particles[index].alpha
        # Update display of springs(rendered as helices)
        for (index, __) in enumerate(self.helices):
            self.helices[index].pos = vector_from(self.springs[index].pos)
            self.helices[index].axis = vector_from(self.springs[index].axis)
            self.helices[index].radius = self.springs[index].radius
            self.helices[index].color = vector_from(self.springs[index].color)
            self.helices[index].opacity = self.springs[index].alpha
        # Update display of pointers(rendered as arrows)
        for (index, __) in enumerate(self.arrows):
            self.arrows[index].pos = vector_from(self.pointers[index].pos)
            self.arrows[index].axis = vector_from(self.pointers[index].axis)
            self.arrows[index].shaftwidth = self.pointers[index].shaftwidth
            self.arrows[index].color = vector_from(self.pointers[index].color)
            self.arrows[index].opacity = self.pointers[index].alpha

    def simulate(self, dt=0.01):
        """
        Simulates a time-step with a step size of dt.
        Parameters
        ----------
        dt: float
            Size of time step taken
        """
        # Set up system if only just starting, in case things have changed since system was created.
        if self.time == 0:
            if self.display_forces:
                self._assign_forces()
            if self.visualize:
                self.create_vis()
        # Make pointers appropriate sizes according to forces on particles.
        if self.display_forces:
            for index, pointer in enumerate(self.pointers):
                pointer.pos = self.particles[index].pos + np.array([0., self.particles[index].radius * 2, 0.])
                pointer.axis = self.particles[index].applied_force(self.particles[index].pos, self.time)
        # Forces on particles depending on the force on them
        for particle in self.particles:
            if not particle.fixed:
                particle.total_force = particle.applied_force(particle.pos, self.time)
                for spring in self.springs:
                    particle.total_force += spring.force_on(particle)
            else:
                particle.total_force = np.array([0., 0., 0.])
        # Update particle positions according to the forces.
        for particle in self.particles:
            particle.prev_pos = copy.deepcopy(particle.pos)
            particle.update(dt)
        # record amplitudes/visualize if required.
        if self.record_amplitudes:
            self._get_amplitudes()
        if self.visualize:
            self.update_vis()

    def run_for(self, time, dt=0.01):
        """
        Run simulation for a certain amount of time(as measured in the system's time).
        Recommended to use this instead of simulate(dt) for most situations.
        Parameters
        ----------
        time: float
            Time for which the simulation will go on for in the system's time
        dt: float
            Time steps taken.
        """
        # Make pointer objects if not already created
        if self.display_forces:
            self._assign_forces()
        # Create visualization if necessary
        if self.visualize:
            self.create_vis()
        # Import vpython if necessary
        if self.visualize and self.visualizer_type == "vpython":
            import vpython
        # Simulate for given time
        while self.time < time:
            if self.visualize and self.visualizer_type == "vpython":
                vpython.rate(150)
            self.simulate(dt)
            if self.stop_on_cycle:
                if self._cycle_completed():
                    break
            self.time += dt

    def _cycle_completed(self):
        """
        Function to see if an oscillation cycle has been completed. Only should work for normal modes.
        """
        if self.time == 0:
            for particle in self.particles:
                particle.initial_v = copy.deepcopy(particle.v)
                particle.initial_pos = copy.deepcopy(particle.pos)
                particle.prev_pos = copy.deepcopy(particle.pos)
        if self.time <= 0.01:
            # Don't expect any oscillations to finish faster than 0.01 seconds. Change if necessary.
            return False
        # If any of the particles have gone past their original position/are on their original positions,
        # and the velocity is in the same direction as originally, then cycle is completed.
        for particle in self.particles:
            if not particle.fixed:
                diff = element_mult(particle.pos - particle.initial_pos,
                    particle.prev_pos - particle.initial_pos)
                if (diff[0] <= 0. and diff[1] <= 0. and diff[2] <= 0.):
                    vel_diff = element_mult(particle.initial_v, particle.v)
                    if vel_diff[0] < 0. or vel_diff[1] < 0. or vel_diff[2] < 0.:
                        return False
                    return True
                return False
        return False

    def _get_amplitudes(self):
        """
        Gets the amplitudes for all the particles system. Only verified to work for 1D oscillations.
        """
        for particle in self.particles:
            vel_diff = element_mult(particle.v, particle.prev_v)
            if vel_diff[0] < 0. or vel_diff[1] < 0. or vel_diff[2] < 0.:
                if not particle.max_point.any():
                    if not particle.min_point.any():
                        particle.max_point = copy.deepcopy(particle.pos)
                        particle.min_point = copy.deepcopy(particle.pos)
                else:
                    if particle.pos[0] > particle.max_point[0]:
                        particle.max_point = copy.deepcopy(particle.pos)
                    else:
                        particle.min_point = copy.deepcopy(particle.pos)
            particle.prev_v = copy.deepcopy(particle.v)

    def _assign_pointers(self):
        """
        Assign Pointers to the forces on each particle.
        """
        for particle in self.particles:
            if not particle._pointer_assigned:
                self.pointers.append(Pointer(pos=particle.pos,
                    axis = particle.applied_force(particle.pos,0)))
 



