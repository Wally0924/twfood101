import os
from PIL import Image

def convert_rgba_to_rgb(folder_path):
    """
    檢查指定資料夾內的所有JPG和PNG圖片，
    將RGBA格式的圖片轉換為RGB格式並保存。
    
    Args:
        folder_path (str): 要處理的資料夾路徑
    """
    # 檢查資料夾是否存在
    if not os.path.isdir(folder_path):
        print(f"錯誤：{folder_path} 不是一個有效的資料夾路徑")
        return
    
    # 計數器
    total_images = 0
    converted_images = 0
    
    # 支援的圖片格式
    supported_formats = ['.jpg', '.jpeg', '.png']
    
    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 檢查檔案是否為支援的圖片格式
        file_ext = os.path.splitext(filename)[1].lower()
        if os.path.isfile(file_path) and file_ext in supported_formats:
            total_images += 1
            
            try:
                # 開啟圖片
                img = Image.open(file_path)
                
                # 檢查是否為RGBA模式
                if img.mode == 'RGBA':
                    # 轉換為RGB
                    rgb_img = img.convert('RGB')
                    
                    # 保存轉換後的圖片
                    rgb_img.save(file_path)
                    
                    converted_images += 1
                    print(f"已轉換：{filename}")
                
                img.close()
            except Exception as e:
                print(f"處理 {filename} 時發生錯誤：{e}")
    
    print(f"\n處理完成！")
    print(f"總共處理的圖片：{total_images}")
    print(f"已轉換的RGBA圖片：{converted_images}")

if __name__ == "__main__":
    # 請修改這裡的路徑為您要處理的資料夾路徑
    folder_to_process = "/home/rvl/Documents/tw_food_101/datasets/tw_food_101/train"
    convert_rgba_to_rgb(folder_to_process)

    folder_to_process = "/home/rvl/Documents/tw_food_101/datasets/tw_food_101/test"
    convert_rgba_to_rgb(folder_to_process)