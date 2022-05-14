import os

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


class DriveExplorer:
    def __init__(self, on_colab):
        self.on_colab = on_colab
        if not on_colab:
            gauth = GoogleAuth()
            gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication.
            self.drive = GoogleDrive(gauth)
            self.TRIS_FOLDER = "1omRd2PjeBoyBWqONqMjkBQSIljeBTtoO"
            self.training_data_folder = "17NUE-oiWUElF_oI6g0j51VSdqWAJS3JQ"
            self.models_folder = "15ImJpOnun-hG1t1JncsUIArXNfi-GSOl"

    def save_file(self, file_name, file_content):
        file1 = self.drive.CreateFile({"title": file_name})
        file1.SetContentString(file_content)
        file1.Upload()
        print(file1["id"])

    def get_file(self, file_name):
        file1 = self.drive.CreateFile({"title": file_name})
        # file1.GetContentFile(file_name)
        return file1.GetContentString()

    def list_dir(self, parent_directory_id):
        return self.drive.ListFile(
            {"q": "'{}' in parents and trashed=false".format(parent_directory_id)}
        ).GetList()

    def get_file_in_folder(self, file_name, parent_directory_id):
        foldered_list = self.list_dir(parent_directory_id)
        for file in foldered_list:
            if file["title"] == file_name:
                return file
        return None

    def create_file_in_folder_by_string(
        self, file_name, parent_directory_id, file_content
    ):
        file1 = self.drive.CreateFile(
            {"title": file_name, "parents": [{"id": parent_directory_id}]}
        )
        file1.SetContentString(file_content)
        file1.Upload()
        return file1["id"]

    def create_file_in_folder_by_file(
        self, file_name, parent_directory_id, source_file
    ):
        file1 = self.drive.CreateFile(
            {"title": file_name, "parents": [{"id": parent_directory_id}]}
        )
        file1.SetContentFile(source_file)
        file1.Upload()
        return file1["id"]

    def retrieve_model(self, model_name):
        if self.on_colab:
            os.popen(
                f"cp -r  '/content/gdrive/My Drive/TRIS/models/{model_name}.pt' './models/'"
            )
        else:
            self.get_file_in_folder(
                f"{model_name}.pt", self.models_folder
            ).GetContentFile(f"./models/{model_name}.pt")

    def save_model(self, model_name):
        if self.on_colab:
            os.popen(
                f"cp -r './models/{model_name}.pt' '/content/gdrive/My Drive/TRIS/models'"
            )
        else:
            self.create_file_in_folder_by_file(
                f"{model_name}.pt", self.models_folder, f"./models/{model_name}.pt"
            )

    def retrieve_training_data(self, data_name):
        if self.on_colab:
            os.popen(
                f"cp -r '/content/gdrive/My Drive/TRIS/training_data/{data_name}_policies.pkl' './training_data/' "
            )
            os.popen(
                f"cp -r '/content/gdrive/My Drive/TRIS/training_data/{data_name}_states.pkl' './training_data/' "
            )
            os.popen(
                f"cp -r '/content/gdrive/My Drive/TRIS/training_data/{data_name}_values.pkl' './training_data/' "
            )
        else:
            self.get_file_in_folder(
                f"{data_name}_policies.pkl", self.training_data_folder
            ).GetContentFile(f"./training_data/{data_name}_policies.pkl")
            self.get_file_in_folder(
                f"{data_name}_states.pkl", self.training_data_folder
            ).GetContentFile(f"./training_data/{data_name}_states.pkl")
            self.get_file_in_folder(
                f"{data_name}_values.pkl", self.training_data_folder
            ).GetContentFile(f"./training_data/{data_name}_values.pkl")

    def save_training_data(self, data_name):
        if self.on_colab:
            os.popen(
                f"cp -r './training_data/{data_name}_policies.pkl' '/content/gdrive/My Drive/TRIS/training_data'"
            )
            os.popen(
                f"cp -r './training_data/{data_name}_states.pkl' '/content/gdrive/My Drive/TRIS/training_data'"
            )
            os.popen(
                f"cp -r './training_data/{data_name}_values.pkl' '/content/gdrive/My Drive/TRIS/training_data'"
            )
        else:
            self.create_file_in_folder_by_file(
                f"{data_name}_policies.pkl",
                self.training_data_folder,
                f"./training_data/{data_name}_policies.pkl",
            )
            self.create_file_in_folder_by_file(
                f"{data_name}_states.pkl",
                self.training_data_folder,
                f"./training_data/{data_name}_states.pkl",
            )
            self.create_file_in_folder_by_file(
                f"{data_name}_values.pkl",
                self.training_data_folder,
                f"./training_data/{data_name}_values.pkl",
            )


if __name__ == "__main__":
    drive = DriveExplorer(on_colab=False)
    drive.retrieve_training_data("FIXED_61")
