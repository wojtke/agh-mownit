from ocr import OCR
import argparse

parser = argparse.ArgumentParser(description='OCR')
parser.add_argument('-i', '--image', help='Path to image', required=True)
parser.add_argument('-f', '--font', help='Path to font', default='fonts/times.ttf')
parser.add_argument('-o', '--output', help='Path to output file', default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    ocr = OCR(args.font)
    text = ocr.scan(args.image)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(text)
    else:
        print(text)
