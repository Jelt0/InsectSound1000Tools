'''
The function fetchfiles will return a list of absolute path's of all files
under the given path.
If a keyword is given as the optional second argument, only the absolute path
of the files containing the keyword in their filename are returned.
Furthermore, keyword_not maybe given as the third optional argument. The
keyword_not may be a longer version of the keyword, that is not supposed
to trigger the return of the path (e.g. keyword=fire, keyword_not=firefly)

                                                                Jelto Branding
'''

import os
import re


def fetchfiles(path, keyword=None, keyword_not=None):
    # Take note of original dir
    owd = os.getcwd()
    # Change to dir specified in path:
    os.chdir(path)
    # Safe current working dic path as string:
    cwd = os.getcwd()
    # Create empty list to store results:
    abspath_list = []
    # Sore all filenames in cwd in a list:
    filenames = os.listdir(cwd)
    # cd bake to original dir:
    os.chdir(owd)

    # If no keyword was given, fetch all file names in path:
    if keyword is None:
        # Build a list of absolute path's by joining all
        # the cwd and filename stings:
        for filename in filenames:
            abspath_list.append(os.path.abspath(os.path.join(cwd, filename)))

        if len(abspath_list) == 0:
            print('The given path contained no files.')

        return abspath_list

    # Else, if keyword was given, check all file names in path if they do
    # containing the keyword and only append if the do:
    else:
        # Check for dots and put them in braces:
        if '.' in keyword:
            keyword = keyword.replace('.', '[.]')
        # search and append
        for filename in filenames:
            if re.search(keyword, filename) is not None:
                # if no keyword_not was given, append:
                if keyword_not is None:
                    abspath_list.append(
                        os.path.abspath(os.path.join(cwd, filename)))
                # else, check if the keyword hit was keyword_not and only
                # append if not:
                else:
                    if re.search(keyword_not, filename) is None:
                        abspath_list.append(
                            os.path.abspath(os.path.join(cwd, filename)))

        if len(abspath_list) == 0:
            print('The given path contained no files matching '
                  'the keyword in the filename.')

        return abspath_list


######################
# Debugging:
#path = './Images_Boat1'
#list = fetchfiles(path, 'Boat')
#for abspath in list:
#    print(abspath)

