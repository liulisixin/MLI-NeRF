from projects.NeuralLumen.scripts.copy_config import replace_and_save_new_file


if __name__ == "__main__":
    # Example usage
    folder_path = '/ghome/yyang/PycharmProjects/neuralangelo/'

    filename = 'job_NRHints_Pikachu_b_1'
    old_string = 'Pikachu'
    new_scene_list = ['Pixiu', 'FurScene']
    for new_scene in new_scene_list:
        new_string = new_scene
        new_filename = filename.replace(old_string, new_scene)
        # Call the function with the example parameters
        replace_and_save_new_file(folder_path, filename, new_filename, old_string, new_string)
