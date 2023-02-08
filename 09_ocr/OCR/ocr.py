
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils import display, debug


class ImagePreprocesor:
    """Responsible for preprocessing the image. Methods can be piped together."""

    def __init__(self):
        self.img = None
        self.est_font_size = None
        self.lines = None

        self.inverted = None

    def read_file(self, filename):
        """
        Reads the image from the file.

        Args:
            filename: The filename of the image.
        """
        self.img = cv2.imread(filename)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.inverted = False
        return self

    def denoise(self, sigma=3, blocksize=21, C=5):
        """
        Denoises the image.

        Args:
            sigma: The sigma of the gaussian blur.
            blocksize: The size of the block for the median filter.
            C: The constant for the threshold.
        """
        self.__assert_coherence(inverted=False)

        blurred = cv2.GaussianBlur(self.img, (sigma, sigma), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_TOZERO)[1]
        self.img = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, C)
        return self

    @staticmethod
    def __sort_clockwise(pts, center=None):
        """
        Sort points clockwise.

        Args:
            pts: The points to sort.
            center: The center of the points.
        Returns:
            The sorted points.
        """
        center = center or pts.mean(axis=0)
        pts = sorted(pts, key=lambda x: np.arctan2(x[0] - center[0], x[1] - center[1]))
        return np.array(pts)

    def center_text_block(self, sigma=41, pad=140):
        """
        Detects the text block in the image and centers it.

        Args:
            sigma: The sigma of the gaussian blur.
            pad: The padding to add to the text block.
        """
        self.__assert_coherence(inverted=False)

        # detect block of text and focus on it
        blurred = cv2.GaussianBlur(self.img, (sigma, sigma), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh == 0))
        (x, y), (w, h), angle = cv2.minAreaRect(coords)
        if angle < 45:
            w, h = h, w
            angle += 90

        # warp
        dst = cv2.boxPoints(((x, y), (w + sigma, h + sigma), angle))[:, [1, 0]]
        dst = self.__sort_clockwise(dst)
        src = cv2.boxPoints(((w / 2, h / 2), (w + sigma, h + sigma), 0))
        src = self.__sort_clockwise(src)
        M = cv2.getPerspectiveTransform(dst, src)
        img = cv2.warpPerspective(self.img, M, np.int32([w, h]))

        # threshold
        img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)[1]

        # pad
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        self.img = img

        return self

    def smooth(self, kernel_size=3):
        """
        Smooths the image using morphological operations.

        Args:
            kernel_size: The size of the kernel.
        """
        self.__assert_coherence(inverted=True)

        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size)))
        return self

    def detect_lines(self, shift=2):
        """
        Detects the lines of text in the image.

        Args:
            shift: The shift of the lines.
        Returns:
            lines (np.ndarray): The lines in the image, list of y-coordinates.
            font_size (float): Estimated font size.
        """
        self.__assert_coherence(inverted=True)

        y = self.img.mean(axis=1)
        y = np.minimum(1, 3 * y / np.max(y))
        y = np.convolve(y, np.hanning(5), 'same')

        dif = y - np.roll(y, -1)
        peaks = (dif > np.roll(dif, -1)) * (dif > np.roll(dif, 1))
        threshold = dif > np.max(dif) / 2

        lines = np.argwhere(threshold * peaks)

        spaces = lines[1:] - np.roll(lines, 1)[1:]
        valid_spaces = np.abs((spaces - spaces.mean()) / (spaces.std() + 0.1) < 1)
        font_size = spaces[valid_spaces].mean()

        self.lines = lines + shift
        self.est_font_size = font_size

    def invert(self):
        """
        Inverts the image.
        """
        self.__assert_coherence()
        self.img = cv2.bitwise_not(self.img)
        self.inverted = not self.inverted
        return self

    def normalize(self):
        """
        Normalizes the image.
        """
        self.__assert_coherence()
        self.img = self.img / 255
        return self

    def __assert_coherence(self, inverted=None):
        if self.img is None:
            raise Exception("No image loaded.")

        if inverted is not None and self.inverted != inverted:
            raise Exception("Image should be inverted." if inverted else "Image should not be inverted.")

    def get_img(self):
        """
        Returns the image.
        """
        self.__assert_coherence()
        return self.img

    def get_font_size(self):
        """
        Returns the estimated font size.
        """
        if self.est_font_size is None:
            raise Exception("Line detections has not been completed.")
        return self.est_font_size

    def get_lines(self):
        """
        Returns the estimated lines.
        """
        if self.lines is None:
            raise Exception("Line detections has not been completed.")
        return self.lines


class CharacterLoader:
    """Responsible for loading and preprocessing character imgs. Methods can be piped together."""

    def __init__(self, font_path, chars):
        self.chars = chars
        self.font_path = font_path
        self.char_imgs = None
        self.char_spans = None

    def load_chars(self, size):
        font = ImageFont.truetype(self.font_path, round(size))
        result = []
        for char in self.chars:
            out = Image.new("P", (round(size), round(size * 1.2)), 255)
            draw = ImageDraw.Draw(out)
            draw.text((4, 0), char, font=font)
            result.append(np.array(out))

        self.char_imgs = cv2.bitwise_not(np.array(result)) / 255
        self._detect_spans()

        return self

    def _detect_spans(self):
        self.char_spans = []
        for char_img, char in zip(self.char_imgs, self.chars):
            horizontal = char_img.sum(axis=0) > 0
            start = np.argmax(horizontal)
            end = np.argmin(horizontal[start:]) + start
            if char in '.,;:?!':
                end += 5

            self.char_spans.append((start, end))

        return self

    def adjust_chars(self):
        for i, (char, char_span) in enumerate(zip(self.char_imgs, self.char_spans)):
            outline = cv2.dilate(char, np.ones((3, 3), dtype=np.uint8), iterations=1)

            bg = np.zeros(char.shape)
            bg[char.shape[0] // 4:5 * char.shape[0] // 6, char_span[0]:char_span[1]] = 1
            bg = cv2.GaussianBlur(bg, (3, 1 + 2 * (char.shape[1] // 4)), 0)

            result = -bg
            result[outline > 0] = 0
            result[char > 0] = 1
            result[result > 0] /= char.sum()
            result[result < 0] /= -result[result < 0].sum()

            self.char_imgs[i] = result

        return self

    def get_chars_imgs(self):
        return self.char_imgs

    def get_spans(self):
        return self.char_spans


class OCR:
    """
    Uses simple convolution to recognize characters in the image. You have to provide font path and character list to
    be detected. The font size is estimated using the lines in the image. It is recommended not to use small font sizes.
    """

    def __init__(self, font_path, chars="abcdefghijklmnopqrstuvwxyz0123456789"):
        self.IP = ImagePreprocesor()
        self.CL = CharacterLoader(font_path, chars)
        self.font_path = font_path

        self.chars_imgs = None
        self.chars_spans = None
        self.chars = chars

    def scan(self, filename):
        """
        Scans the image and returns the text.

        Args:
            filename (str): The filename of the image.
        Returns:
            text (str): The detected text in the image.
        """
        img, lines, font_size = self._load_img(filename)
        char_imgs, char_spans = self._load_chars(font_size)
        corrs = self.get_corrs(img, char_imgs)
        corrs = self.linearize_corrs(corrs, lines)
        spaces = self.detect_spaces(img, font_size, lines)
        detections = self.pick_detections(corrs, char_spans, spaces)
        text = self.get_text(detections)
        return text

    def _load_chars(self, size):
        """
        Loads characted imgs.

        Args:
            font_path (str): The path to the font.
            size (int): The font size.
            chars (str): The characters to load.
        Returns:
            char_imgs (list): The loaded character imgs.
            char_spans (list): The character spans.
        """

        self.CL.load_chars(size).adjust_chars()
        self.chars_imgs = self.CL.get_chars_imgs()
        self.chars_spans = self.CL.get_spans()
        return self.chars_imgs, self.chars_spans

    def _load_img(self, filename):
        """
        Loads and preprocesses the image.

        Args:
            filename (str): The filename of the image.
        Returns:
            img (np.array): The image.
            lines (list): Detected lines.
            font_size (int): Estimated font size.
        """
        self.IP.read_file(filename)
        self.IP.denoise().center_text_block().invert().normalize().smooth()
        self.IP.detect_lines()
        img = self.IP.get_img()
        lines = self.IP.get_lines()
        font_size = self.IP.get_font_size()
        return img, lines, font_size

    def get_corrs(self, img, chars):
        """
        Uses convolution to match each of chars with img, returns the correlations.

        Args:
            img (np.array): Preprocessed image.
            chars (list): Preprocessed character imgs.

        Returns:
            corrs (np.array): The correlations. The shape is (chars_num, img_y, img_x).
        """
        img_fft = np.fft.fft2(img)
        result = []
        for c in chars:
            c_fft = np.fft.fft2(np.rot90(c, 2), s=img_fft.shape)
            corr = np.real(np.fft.ifft2(img_fft * c_fft))
            result.append(corr)
        return np.array(result)

    def linearize_corrs(self, corrs, lines, line_window=np.ones(21)):
        """
        Knowns the lines, narrows the correlation map just to the lines and around them.

        Args:
            corrs (np.array): The correlation map. The shape is (chars_num, img_y, img_x).
            lines (list): Detected lines.
            line_window (np.array): The window around the lines to take into account.
        Returns:
            corrs (np.array): The linearized correlation map. The shape is (chars_num, lines_num, img_x).
        """
        linear_corrs = np.zeros((corrs.shape[0], len(lines), corrs.shape[2]))
        for i, corr_map in enumerate(corrs):
            for j, line in enumerate(lines):
                indicies = line + np.arange(len(line_window))
                indicies = indicies[(0 <= indicies) * (indicies < corrs.shape[1])]
                linear_corr = corr_map[indicies, :]
                linear_corr *= line_window.reshape(-1, 1)
                linear_corr = linear_corr.max(axis=0)
                linear_corr = np.convolve(linear_corr, np.ones(5), 'same') / 5
                linear_corrs[i, j, :] = linear_corr
        return np.array(linear_corrs)

    def detect_spaces(self, img, char_size, lines):
        """
        Detects the spaces in the image.

        Args:
            img (np.array): The image.
            char_size (int): The character size.
            lines (list): Detected lines.

        Returns:
            spaces (np.array): Array with 1s where the whitespace (spaces) are and 0s where not.
                The shape is (lines_num, img_x).
        """
        space = np.ones((int(char_size // 2), int(char_size // 3)))
        space /= space.sum()
        space_corr = self.get_corrs(img, [space])
        space_corr = self.linearize_corrs(space_corr, lines)[0]
        for i in range(len(space_corr)):
            space_corr[i] = np.convolve(space_corr[i], np.ones(int(char_size // 3)), 'same') / int(char_size // 3)
        return space_corr < 0.2

    def pick_detections(self, corrs, char_spans, spaces):
        """
        Based on linearized correlation map, detected spaces and how wide the character is,
        picks the best points to consider detected character (by max corr value at given place).
        """
        occupied = np.copy(spaces).astype(np.int32)  # TODO bool ?
        picks = np.copy(spaces).astype(np.int32)
        mean = corrs.mean()  # TODO use sth else

        for i in range(corrs.shape[1]):
            line = corrs[:, i, :]

            best_scores = np.max(line, axis=0)
            best_chars = np.argmax(line, axis=0)
            best_spots = np.argsort(-best_scores)

            for spot in best_spots:
                score = best_scores[spot]
                char = best_chars[spot]
                if score < mean:
                    break
                start, end = char_spans[char]
                if not np.any(occupied[i, spot + start - end:spot] > 0):
                    picks[i, spot + start - end] = char + 2
                    occupied[i, spot + start - end + 2:spot - 2] = 2

        return picks

    def get_text(self, detections):
        """
        Converts the detections point list to text.
        """
        char_arr_new = ' ' + self.chars
        output = ''
        for line in detections:
            chars = [char_arr_new[x] for x in line[line > 0].astype(np.int32) - 1]
            output += ' '.join(''.join(chars).split()) + '\n'

        return output

