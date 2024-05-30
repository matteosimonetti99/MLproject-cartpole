
import math
import random
from typing import Optional, Tuple, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from gym.vector.utils import batch_space



class CartPoleModifiedEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                0
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None


        #EDIT
        self.laser0X = (self.screen_width // 2)
        self.laser1X = (self.screen_width * 1) // 5          # coordinate X laser sx
        self.laser2X = (self.screen_width * 4) // 5         # coordinate X laser dx
        self.laserY = self.screen_height                    # coordinate Y laser mid
        self.laserSize = 20                                 # used to draw the laser starting points
        self.laser0On = False       # start value of laser 0 (if it is changed here must be changed in reset() too)
        self.laser12On = False       # start value of laser 1 and 2 (if it is changed here must be changed in reset() too)
        self.laserStepsToSwitch = 45    # after how many steps the lasers switch from on to off and viceversa
        self.laserCounter = 0           # to count the steps
        #END OF EDIT


    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state[:4]                         # EDIT
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot


        #EDIT
            
        # Toggle laser:
        self.laserCounter += 1
        if (self.laserCounter % self.laserStepsToSwitch == 0):
            self.laser0On = not self.laser0On
            self.laser12On = not self.laser12On
            if (self.laserCounter >= 2*self.laserStepsToSwitch):
                self.laserCounter = 0
            

        counterNormalized = self.laserCounter/float(self.laserStepsToSwitch) - 1        # now is between -1 and 1, instead of 0 and 2*self.laserStepsToSwitch
            
        #if self.laser0On:
        #    self.state = (x, x_dot, theta, theta_dot, 1, counterNormalized)
        #else:
        #    self.state = (x, x_dot, theta, theta_dot, -1, counterNormalized)
        
        self.state = (x, x_dot, theta, theta_dot, counterNormalized)

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        cart_sx = (x * scale + self.screen_width / 2.0) - 50/2    # left border of the cart (formula taken from below)
        cart_dx = (x * scale + self.screen_width / 2.0) + 50/2    # right border of the cart

        
        hitLasers = bool((self.laser0On and cart_sx <= self.laser0X and cart_dx >= self.laser0X)
                      or (self.laser12On and cart_sx <= self.laser1X and cart_dx >= self.laser1X)
                      or (self.laser12On and cart_sx <= self.laser2X and cart_dx >= self.laser2X))
        
        #END OF EDIT


        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or hitLasers                                #EDIT
        )

        if not terminated:
            reward = 1.0
        elif hitLasers:
            reward=-5.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = -55.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undeEND OFd behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}     # EDIT (added hitLasers)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        
        self.laser0On = False   #EDIT: laser starting value
        self.laser12On = False   #EDIT: laser starting value
        self.laserCounter = 0
        
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        #self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.state = ( *self.np_random.uniform(low=low, high=high, size=(4,)), 0)    # EDIT (row above)
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART                (<- formula copied above)
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))


        #EDIT

        # Laser draw:
        if self.laser0On:
            gfxdraw.vline(self.surf, self.laser0X, self.laserY - self.laserSize, int(carty - cartheight/2), (0,255,0))
        
        if self.laser12On:
            gfxdraw.vline(self.surf, self.laser1X, self.laserY - self.laserSize, int(carty - cartheight/2), (0,255,0))
            gfxdraw.vline(self.surf, self.laser2X, self.laserY - self.laserSize, int(carty - cartheight/2), (0,255,0))       
        
        
        # laser boxes draw:
        laser0A = (self.laser0X - self.laserSize + 5, self.laserY)
        laser0B = (self.laser0X - self.laserSize + 5, self.laserY - self.laserSize)
        laser0C = (self.laser0X + self.laserSize - 5, self.laserY - self.laserSize)
        laser0D = (self.laser0X + self.laserSize - 5, self.laserY)
        laser0Coords = [laser0A, laser0B, laser0C, laser0D]
        gfxdraw.filled_polygon(self.surf, laser0Coords, (70, 70, 70))
        
        laser1A = (self.laser1X - self.laserSize + 5, self.laserY)
        laser1B = (self.laser1X - self.laserSize + 5, self.laserY - self.laserSize)
        laser1C = (self.laser1X + self.laserSize - 5, self.laserY - self.laserSize)
        laser1D = (self.laser1X + self.laserSize - 5, self.laserY)
        laser1Coords = [laser1A, laser1B, laser1C, laser1D]
        gfxdraw.filled_polygon(self.surf, laser1Coords, (70, 70, 70))

        laser2A = (self.laser2X - self.laserSize + 5, self.laserY)
        laser2B = (self.laser2X - self.laserSize + 5, self.laserY - self.laserSize)
        laser2C = (self.laser2X + self.laserSize - 5, self.laserY - self.laserSize)
        laser2D = (self.laser2X + self.laserSize - 5, self.laserY)
        laser2Coords = [laser2A, laser2B, laser2C, laser2D]
        gfxdraw.filled_polygon(self.surf, laser2Coords, (70, 70, 70))

        #END OF EDIT


        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


        

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False



