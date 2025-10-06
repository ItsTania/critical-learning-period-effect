import torch
from torchvision import transforms
from typing import Optional

transform_3ch = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    ])

COLOUR_MAP = {
            0: ('red',     [1.0, 0.2, 0.2]),
            1: ('green',   [0.2, 1.0, 0.2]),
            2: ('blue',    [0.2, 0.2, 1.0]),
            3: ('yellow',  [1.0, 1.0, 0.2]),
            4: ('cyan',    [0.2, 1.0, 1.0]),
            5: ('magenta', [1.0, 0.2, 1.0]),
            6: ('orange',  [1.0, 0.6, 0.2]),
            7: ('purple',  [0.6, 0.2, 1.0]),
            8: ('lime',    [0.6, 1.0, 0.2]),
            9: ('brown',   [0.6, 0.4, 0.2])
        }

class ColorMNIST(torch.utils.data.Dataset):
    def __init__(self,
                 mnist_dataset,
                 theta: Optional[float] = None,
                 fixed_colour_ind: Optional[int] = None,
                 colour_map = COLOUR_MAP
                 ):

        self.mnist = mnist_dataset
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        self.theta = theta
        self.fixed_colour_ind = fixed_colour_ind
        self.shape_colour_map = colour_map
        self.color_map_tensor = {
            k: torch.tensor(v[1], dtype=torch.float32).view(3,1,1)
            for k,v in self.shape_colour_map.items()
            }
        self.num_classes = len(self.shape_colour_map)

    def __getitem__(self, index):
        '''
        Get an item from the dataset, applying the spurious colour logic
        Returns a tuple of (image, digit, colour)
        '''
        img, label = self.mnist[index]

        # Colour assignment
        if self.fixed_colour_ind is not None:
            colour_int = self.fixed_colour_ind # Fixed
        else:
            probs = self.get_colour_probabilities(colour_index=int(label))
            colour_int = torch.multinomial(probs, 1).item()

        # Transform greyscale image to RGB
        img_tensor = self.transform(img)
        colorised_img = img_tensor * self.color_map_tensor[colour_int]

        return colorised_img, label #, colour_int

    def get_colour_name(self, colour_index):
        assert colour_index in self.shape_colour_map, \
            f"Colour index must be one of {list(self.shape_colour_map.keys())}, got {colour_index}"
        return self.shape_colour_map[colour_index][0]

    def get_colour_rgb(self, colour_index):
        assert colour_index in self.shape_colour_map, \
            f"Colour index must be one of {list(self.shape_colour_map.keys())}, got {colour_index}"
        return self.shape_colour_map[colour_index][1]

    def get_colour_probabilities(self, colour_index):
      identity = torch.zeros(self.num_classes)
      identity[colour_index] = 1.0

      uniform = torch.ones(self.num_classes) / self.num_classes

      probs = self.theta * identity + (1 - self.theta) * uniform
      return probs

    def update_theta(self, new_theta):
      self.theta = new_theta

    def __len__(self):
        return len(self.mnist)
    