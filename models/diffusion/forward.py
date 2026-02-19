
def noise_schedule(t: int, T: int) -> float:
    # TODO: implement a noise schedule function that returns the amount of noise to add at time step t
    pass

def forward(x, T, t_max, noise_schedule: callable[[int, int], float], one_shot: bool=False):
    # T_max = int(pct_T * T) -> stop prematurely
    # TODO: add noise to x according to the forward diffusion process for t<T time steps
    pass


if __name__ == '__main__':
    # TODO: test the forward function with an image (see how gaussian noise is added to the image with increasing time steps/one shot)
    pass