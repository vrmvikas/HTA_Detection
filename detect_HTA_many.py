from utils import *
import sys
GT_path = sys.argv[1]
import os

def clear_screen():
    """Clears the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")

parameters = {
    "MIN_AREA_THRESHOLD": 3,
    "BoundryColor": (255,0,0),
    "BinThreshold": 50,
    "Num_Of_Detections": 50
}


# MIN_AREA_THRESHOLD = parameters["MIN_AREA_THRESHOLD"]
# BoundryColor = parameters["BoundryColor"]
# BinThreshold = parameters["BinThreshold"]
# Num_Of_Detections = parameters["Num_Of_Detections"]


def Process_images(parameters):    
    for idx, testImagePath in enumerate(sys.argv[2:]):    
        testImage, GT, BinImage, contours = Get_Contours_Of_SusRegions(testImagePath, GT_path, BinThreshold= parameters["BinThreshold"], MIN_AREA_THRESHOLD = parameters["MIN_AREA_THRESHOLD"])
        All_distances = SiameseNetwork(contours, testImage, GT)
        # threshold = 1
        cleanContours = [contours[i] for i in np.argsort(All_distances)[::-1][:parameters["Num_Of_Detections"]]]    
        modifiedImage = Draw_Rectangle_Using_xywh(testImage, cleanContours, thickness=5, color=parameters["BoundryColor"])
        plt.imsave("op1.jpg", modifiedImage)
        plt.imsave("op_" + str(idx) +".jpg", Draw_Rectangle_Using_xywh(np.ones_like(GT)*255, cleanContours, thickness=5, color=parameters["BoundryColor"]))


def currentParameters(parameters):
    return f'Boundry Color: {parameters["BoundryColor"]}\nNumber of Detections: {parameters["Num_Of_Detections"]}\nBinarisation Threshold: {parameters["BinThreshold"]}\nMinimum Area of detection box: {parameters["MIN_AREA_THRESHOLD"]}'


def main():
    while True:
        print("Current Parameters:\n")
        print(currentParameters(parameters))
        print("\n\nCommands:\nY:\tProcess the inputs\nClear:\tClears the screen\nC:\tChange the Parameters")
        user_input = input("Enter a command (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting the application. Goodbye!")
            break
        elif user_input.lower() == 'y':
            Process_images(parameters)
        elif user_input.lower() == "clear":
            clear_screen()
        elif user_input.lower() == 'c':
            while True:
                clear_screen()
                print("Current Parameters:\n")
                print(currentParameters(parameters))
                parameter_keys = list(parameters.keys())
                for i, para in enumerate(parameter_keys):
                    print(i,'\t', parameter_keys[i])
                choice = input(f"Enter parameter to change: {0} to {len(parameter_keys)-1}, Enter \"exit\" if you are done")
                if choice.lower() == 'exit':
                    break            
                ans = int(input("Enter new value"))
                parameters[parameter_keys[int(choice)]] = ans
                clear_screen()            
        else:
            continue
    
if __name__ == "__main__":
    main()