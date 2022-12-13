from preprocessor import Preprocessor

if __name__ == '__main__':
    input_directory = '../../data/raw/test_images/'
    output_directory = '../../data/processed/test_images_processed/'
    print(f'Preprocessing: {input_directory}')
    preprocessor = Preprocessor(input_directory=input_directory, output_directory=output_directory, workers=32)
    preprocessor.preprocess()
