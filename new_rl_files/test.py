import unittest
import numpy as np
from utils import plot_trajectory

class TestPlotting(unittest.TestCase):
    def test_plot_trajectory(self):
        # Define the grid, trajectory, and conf for testing
        grid = np.array([[1, 2], [3, 4]])
        trajectory = [(0, 0), (1, 1)]
        conf = {'start': [0, 0], 'end': [1, 1]}

        # Call the function to test
        plot_trajectory(grid, trajectory, conf)

        # Since it's a plotting function, we can't assert the output directly.
        # But we can check if the function runs without errors.
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()