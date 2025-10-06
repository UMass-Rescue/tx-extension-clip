from unittest.mock import MagicMock

MOCKED_MODULES = {
    "torch": MagicMock(),
    "torch.nn": MagicMock(),
    "torch.nn.functional": MagicMock(),
    "open_clip": MagicMock(),
    "open_clip.factory": MagicMock(),
    "torchvision": MagicMock(),
    "torchvision.transforms": MagicMock(),
}
