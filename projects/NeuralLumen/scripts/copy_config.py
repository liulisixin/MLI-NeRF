import os


def replace_and_save_new_file(folder_path, filename, new_filename, old_string, new_string):
    """
    Replaces occurrences of a specified string in a file with a new string and saves the result to a new file.

    Parameters:
    - folder_path: The path to the folder containing the original file.
    - filename: The name of the original file.
    - new_filename: The name for the new file with replacements.
    - old_string: The string in the file to be replaced.
    - new_string: The string to replace the old_string with.
    """
    # Create the full path to the original and new file
    file_path = os.path.join(folder_path, filename)
    new_file_path = os.path.join(folder_path, new_filename)

    # Attempt to open the original file in read mode and the new file in write mode
    try:
        with open(file_path, 'r', encoding='utf-8') as file, open(new_file_path, 'w', encoding='utf-8') as new_file:
            # Read the content of the original file
            content = file.read()
            count_replacements = content.count(old_string)
            # Replace the old string with the new string
            new_content = content.replace(old_string, new_string)
            # Write the modified content to the new file
            new_file.write(new_content)

        print(f"File '{filename}' has been processed. New file '{new_filename}' created with {count_replacements} replacements.")
    except FileNotFoundError:
        print(f"File '{filename}' not found in '{folder_path}'. Please check the file name and path.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    """
    Pay attention to:
    1   scale
    2   sphere_tracing_num_iter
    """
    # old_string = 'garden'
    # new_scene_list = ['apple',
    #                   'savannah',
    #                   'cube']
    # old_string = 'hotdog'
    # new_scene_list = ['lego',
    #                   'FurBall',
    #                   'drums']
    old_string = 'Pikachu'
    new_scene_list = ['Fish',
                      'FurScene',
                      'Pixiu']
    #
    # folder_path = '/ghome/yyang/PycharmProjects/neuralangelo/projects/NeuralLumen/configs'
    # filename = 'syn_hotdog_NL4_b.yaml'
    # for new_scene in new_scene_list:
    #     new_string = new_scene
    #     new_filename = filename.replace(old_string, new_scene)
    #     # Call the function with the example parameters
    #     replace_and_save_new_file(folder_path, filename, new_filename, old_string, new_string)

    folder_path = '/ghome/yyang/PycharmProjects/NRHints/'
    filename = 'job_test_Pikachu'
    for new_scene in new_scene_list:
        new_string = new_scene
        new_filename = filename.replace(old_string, new_scene)
        # Call the function with the example parameters
        replace_and_save_new_file(folder_path, filename, new_filename, old_string, new_string)

    # old_string = 'NL1'
    # new_string = 'NL4'
    # folder_path = '/ghome/yyang/PycharmProjects/neuralangelo/projects/NeuralLumen/configs'
    # filename_list = ['syn_FurBall_NL1_a.yaml', 'syn_FurBall_NL1_b.yaml',
    #             'syn_drums_NL1_a.yaml', 'syn_drums_NL1_b.yaml',
    #             'syn_lego_NL1_a.yaml', 'syn_lego_NL1_b.yaml']
    # for filename in filename_list:
    #     new_filename = filename.replace(old_string, new_string)
    #     # Call the function with the example parameters
    #     replace_and_save_new_file(folder_path, filename, new_filename, old_string, new_string)
