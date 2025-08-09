import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk
import threading

class WatermarkRemover:
    def __init__(self):
        self.debug = False
    
    def set_debug(self, debug):
        """设置是否显示调试信息和中间结果"""
        self.debug = debug

    def detect_watermark(self, image):
        """检测图像中可能的水印区域 - 增强版"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        
        # 创建水印掩码
        watermark_mask = np.zeros_like(gray)
        possible_watermarks = []
        
        # 方法1: 检测右下角区域 - 这是水印的常见位置
        roi_h, roi_w = int(h * 0.25), int(w * 0.25)  # 扩大检查区域到右下25%
        roi_y, roi_x = h - roi_h, w - roi_w
        roi = gray[roi_y:, roi_x:]
        
        # 对ROI应用多种阈值处理方法
        thresholds = [
            cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)[1],  # 亮色水印
            cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)[1],       # 暗色水印
            cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ]
        
        for thresh in thresholds:
            # 使用形态学操作改善水印区域
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 寻找轮廓
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 筛选可能的水印区域
            for contour in contours:
                area = cv2.contourArea(contour)
                if 20 < area < 8000:  # 扩大面积范围，降低下限以捕获更小的水印
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.1 < aspect_ratio < 10:  # 放宽宽高比限制
                        # 将坐标转换回原图坐标系
                        global_x, global_y = x + roi_x, y + roi_y
                        possible_watermarks.append((global_x, global_y, w, h))
                        # 在掩码上绘制轮廓
                        cv2.drawContours(watermark_mask[roi_y:, roi_x:], [contour], -1, 255, -1)
        
        # 方法2: 检测整个图像中的特定颜色区域 (针对彩色水印)
        # 转换为HSV颜色空间以更好地检测颜色
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 检测多种颜色范围
        color_ranges = [
            # 蓝色范围
            (np.array([100, 50, 50]), np.array([130, 255, 255])),
            # 红色范围1 (Hue值在0-10)
            (np.array([0, 50, 50]), np.array([10, 255, 255])),
            # 红色范围2 (Hue值在170-180)
            (np.array([170, 50, 50]), np.array([180, 255, 255])),
            # 白色范围
            (np.array([0, 0, 200]), np.array([180, 30, 255])),
            # 黑色范围
            (np.array([0, 0, 0]), np.array([180, 255, 30]))
        ]
        
        color_mask = np.zeros_like(gray)
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            color_mask = cv2.bitwise_or(color_mask, mask)
        
        # 使用形态学操作改善掩码
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选可能的水印区域
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 50000:  # 对于颜色检测，使用更大的面积范围
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.1 < aspect_ratio < 10:  # 放宽宽高比限制
                    possible_watermarks.append((x, y, w, h))
                    # 在掩码上绘制轮廓
                    cv2.drawContours(watermark_mask, [contour], -1, 255, -1)
        
        # 方法3: 检测图像各个区域的小型水印
        # 将图像分为多个区域进行检测
        regions = [
            # 右下角
            (w - int(w * 0.25), h - int(h * 0.25), w, h),
            # 右上角
            (w - int(w * 0.25), 0, w, int(h * 0.25)),
            # 左下角
            (0, h - int(h * 0.25), int(w * 0.25), h),
            # 中心区域
            (int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))
        ]
        
        for x1, y1, x2, y2 in regions:
            region = gray[y1:y2, x1:x2]
            if region.size == 0:  # 跳过空区域
                continue
                
            # 增强对比度
            region_enhanced = cv2.equalizeHist(region)
            
            # 应用Canny边缘检测
            edges = cv2.Canny(region_enhanced, 30, 150)
            
            # 使用形态学操作连接边缘
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # 寻找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 筛选可能的水印区域
            for contour in contours:
                area = cv2.contourArea(contour)
                if 10 < area < 3000:  # 针对小型水印
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.1 < aspect_ratio < 10:  # 放宽宽高比限制
                        # 将坐标转换回原图坐标系
                        global_x, global_y = x + x1, y + y1
                        possible_watermarks.append((global_x, global_y, w, h))
                        # 在掩码上绘制轮廓
                        temp_mask = np.zeros_like(watermark_mask)
                        cv2.rectangle(temp_mask, (global_x, global_y), (global_x+w, global_y+h), 255, -1)
                        watermark_mask = cv2.bitwise_or(watermark_mask, temp_mask)
        
        # 方法4: 特别针对图片中的红色绳子和小型水印
        # 提取红色通道
        red_channel = image[:,:,2]
        
        # 对红色通道应用阈值处理
        _, red_thresh = cv2.threshold(red_channel, 150, 255, cv2.THRESH_BINARY)
        
        # 寻找红色区域的轮廓
        contours, _ = cv2.findContours(red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选可能的红色水印区域
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 1000:  # 针对小型红色元素
                x, y, w, h = cv2.boundingRect(contour)
                possible_watermarks.append((x, y, w, h))
                # 在掩码上绘制轮廓
                cv2.drawContours(watermark_mask, [contour], -1, 255, -1)
        
        # 使用连通区域分析，移除太小的区域（噪点）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(watermark_mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 10:  # 降低阈值以保留更多小区域
                watermark_mask[labels == i] = 0
        
        # 扩展水印区域以确保完全覆盖
        kernel = np.ones((3, 3), np.uint8)
        watermark_mask = cv2.dilate(watermark_mask, kernel, iterations=1)
        
        if self.debug:
            debug_img = image.copy()
            for x, y, w, h in possible_watermarks:
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Detected Watermark Areas', debug_img)
            cv2.imshow('Watermark Mask', watermark_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return possible_watermarks, watermark_mask
    
    def remove_watermark(self, image, watermark_mask):
        """使用修复算法去除水印 - 增强版"""
        # 创建一个掩码，标记需要修复的区域
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[watermark_mask > 0] = 255
        
        # 计算掩码区域大小
        mask_area = np.sum(mask > 0)
        print(f"水印掩码大小: {mask_area} 像素")
        
        # 安全检查：如果掩码区域过大，可能是误检
        if mask_area > image.shape[0] * image.shape[1] * 0.3:  # 如果超过图像面积的30%
            print("检测到的水印区域过大，可能是误检，正在限制处理区域...")
            # 只保留最可能的水印区域
            h, w = image.shape[:2]
            
            # 方法1: 保留中心区域的掩码
            center_x, center_y = w // 2, h // 2
            center_roi_size = min(w, h) // 3
            center_mask = np.zeros_like(mask)
            cv2.rectangle(center_mask, 
                         (center_x - center_roi_size//2, center_y - center_roi_size//2),
                         (center_x + center_roi_size//2, center_y + center_roi_size//2),
                         255, -1)
            center_mask = cv2.bitwise_and(mask, center_mask)
            
            # 方法2: 保留右下角区域的掩码
            corner_mask = np.zeros_like(mask)
            roi_h, roi_w = int(h * 0.25), int(w * 0.25)
            roi_y, roi_x = h - roi_h, w - roi_w
            corner_mask[roi_y:, roi_x:] = mask[roi_y:, roi_x:]
            
            # 选择非空的掩码
            if np.sum(center_mask > 0) > 0:
                mask = center_mask
            elif np.sum(corner_mask > 0) > 0:
                mask = corner_mask
            else:
                # 如果两种方法都没有找到有效区域，则保留原始掩码的一小部分
                kernel = np.ones((20, 20), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
            
            print(f"调整后的水印掩码大小: {np.sum(mask > 0)} 像素")
        
        # 如果掩码为空，返回原图
        if np.sum(mask > 0) == 0:
            print("未检测到有效水印区域")
            return image
        
        # 扩展掩码区域，确保完全覆盖水印
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 使用多种修复算法填充水印区域
        # 1. 使用TELEA算法
        result_telea = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
        
        # 2. 使用NS算法
        result_ns = cv2.inpaint(image, mask, 5, cv2.INPAINT_NS)
        
        # 3. 结合两种算法的结果
        # 对于复杂纹理区域，NS算法通常效果更好
        # 对于平滑区域，TELEA算法通常效果更好
        # 计算局部方差作为复杂度度量
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.boxFilter(np.float32(gray)**2, -1, (5, 5)) - cv2.boxFilter(np.float32(gray), -1, (5, 5))**2
        variance = cv2.normalize(variance, None, 0, 1, cv2.NORM_MINMAX)
        
        # 根据局部方差选择算法
        # 创建一个三通道的权重掩码
        weight = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            weight[:,:,i] = variance
        
        # 确保权重在0-1范围内
        weight = np.clip(weight, 0, 1)
        
        # 结合两种算法的结果
        result = np.uint8(result_ns * weight + result_telea * (1 - weight))
        
        # 确保只修改掩码区域，其他区域保持不变
        mask_3d = np.zeros_like(image)
        for i in range(3):  # 对BGR三个通道分别处理
            mask_3d[:,:,i] = mask
        
        # 使用位运算而不是浮点数乘法
        # 将掩码区域设为修复后的结果，非掩码区域保持原图
        result = np.where(mask_3d > 0, result, image)
        
        # 确保结果是uint8类型
        if result.dtype != np.uint8:
            result = result.astype(np.uint8)
        
        print(f"处理前图像形状: {image.shape}, 处理后图像形状: {result.shape}, 数据类型: {result.dtype}")
        
        if self.debug:
            cv2.imshow('Original', image)
            cv2.imshow('Mask', mask)
            cv2.imshow('Result', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result

    def process_image(self, input_path, output_path=None):
        """处理图像，检测并去除水印
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径，如果为None则不保存
            
        Returns:
            (success, result): 成功则返回(True, 处理后的图像)，失败则返回(False, 错误信息)
        """
        try:
            # 读取图像 - 使用PIL和numpy处理中文路径
            try:
                import numpy as np
                from PIL import Image
                pil_img = Image.open(input_path)
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                return False, f"无法读取图像: {str(e)}"
            
            # 检测水印
            possible_watermarks, watermark_mask = self.detect_watermark(image)
            
            # 如果检测到的水印区域太小，可能是误检
            if np.sum(watermark_mask > 0) < 30:  # 少于30个像素
                return False, "未检测到明显水印"
            
            # 去除水印
            result = self.remove_watermark(image, watermark_mask)
            
            # 保存结果（如果提供了输出路径）
            if output_path:
                try:
                    # 使用PIL保存，避免中文路径问题
                    pil_result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                    pil_result.save(output_path)
                except Exception as e:
                    return False, f"保存图像失败: {str(e)}"
            
            return True, result
        except Exception as e:
            return False, f"处理图像时出错: {str(e)}"

class WatermarkRemoverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("图片去水印工具")
        self.geometry("900x600")
        self.minsize(800, 500)
        
        self.remover = WatermarkRemover()
        self.input_path = None
        self.output_path = None
        self.original_image = None
        self.processed_image = None
        
        self.create_widgets()
        self.create_menu()
    
    def create_menu(self):
        menu_bar = tk.Menu(self)
        
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="打开图片", command=self.open_image)
        file_menu.add_command(label="保存结果", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.quit)
        
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        
        menu_bar.add_cascade(label="文件", menu=file_menu)
        menu_bar.add_cascade(label="帮助", menu=help_menu)
        
        self.config(menu=menu_bar)
    
    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左右分栏
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 左侧 - 原始图片
        ttk.Label(left_frame, text="原始图片").pack(pady=5)
        self.original_canvas = tk.Canvas(left_frame, bg="#f0f0f0", highlightthickness=1, highlightbackground="#cccccc")
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右侧 - 处理后图片
        ttk.Label(right_frame, text="处理后图片").pack(pady=5)
        self.processed_canvas = tk.Canvas(right_frame, bg="#f0f0f0", highlightthickness=1, highlightbackground="#cccccc")
        self.processed_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 底部控制区域
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="打开图片", command=self.open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="去除水印", command=self.process_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="保存结果", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        # 进度条
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=5)
        
        # 状态标签
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side=tk.RIGHT, padx=5)
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
        
        self.input_path = file_path
        self.status_var.set(f"已加载图片: {os.path.basename(file_path)}")
        
        # 修改这里：使用numpy先读取文件，再转换为OpenCV格式
        try:
            # 方法1：使用numpy读取，再转换为OpenCV格式
            import numpy as np
            from PIL import Image
            pil_img = Image.open(file_path)
            self.original_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # 如果上面方法不行，可以尝试方法2
            # 方法2：直接使用OpenCV，但处理中文路径
            # self.original_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            self.display_image(self.original_image, self.original_canvas, "原始图片")
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
            return
        
        # 清除处理后的图片
        self.processed_image = None
        self.processed_canvas.delete("all")
        self.processed_canvas.create_text(
            self.processed_canvas.winfo_width() // 2,
            self.processed_canvas.winfo_height() // 2,
            text="点击'去除水印'按钮处理图片",
            fill="#999999"
        )
    
    def process_image(self):
        """处理当前加载的图像（按钮回调方法）"""
        if not hasattr(self, 'input_path') or not self.input_path:
            messagebox.showinfo("提示", "请先打开一张图片")
            return
        
        # 显示进度条
        self.progress.start()
        self.status_var.set("正在处理...")
        
        # 在新线程中处理图像
        threading.Thread(target=self._process_image_thread).start()
    
    def _process_image_thread(self):
        try:
            # 创建临时输出路径
            temp_output = os.path.join(os.path.dirname(self.input_path), 
                                     f"temp_{os.path.basename(self.input_path)}")
            
            # 处理图片 - 修改这里，直接获取处理结果而不是通过文件读取
            # 修改WatermarkRemover.process_image方法，使其返回图像数据
            success, result_or_message = self.remover.process_image(self.input_path, temp_output)
            
            if success:
                # 如果result_or_message是字符串(文件路径)，则需要读取文件
                if isinstance(result_or_message, str):
                    try:
                        # 使用PIL读取图像，避免中文路径问题
                        from PIL import Image
                        import numpy as np
                        pil_img = Image.open(result_or_message)
                        self.processed_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        # 如果读取失败，尝试直接从WatermarkRemover获取处理后的图像
                        self.after(0, lambda: messagebox.showerror("错误", f"读取处理后图像失败: {str(e)}\n尝试修改代码以直接返回图像数据"))
                        self.after(0, lambda: self.status_var.set("处理失败"))
                        return
                else:
                    # 如果直接返回的是图像数据，直接使用
                    self.processed_image = result_or_message
                    
                self.output_path = temp_output
                
                # 在主线程中更新UI
                self.after(0, lambda: self.display_image(self.processed_image, self.processed_canvas, "处理后图片"))
                self.after(0, lambda: self.status_var.set("处理完成"))
            else:
                self.after(0, lambda: messagebox.showinfo("信息", f"处理失败: {result_or_message}"))
                self.after(0, lambda: self.status_var.set("处理失败"))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("错误", f"处理过程中出错: {str(e)}"))
            self.after(0, lambda: self.status_var.set("处理出错"))
        finally:
            # 停止进度条
            self.after(0, self.progress.stop)
    
    # 在save_image方法中，修改图像保存部分
    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("警告", "请先处理图片！")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存图片",
            defaultextension=".jpg",
            filetypes=[("JPEG图片", "*.jpg"), ("PNG图片", "*.png"), ("所有文件", "*.*")],
            initialfile=f"{Path(self.input_path).stem}_no_watermark.jpg"
        )
        
        if not file_path:
            return
        
        # 修改这里：使用numpy和OpenCV的imdecode保存带中文路径的图片
        try:
            # 方法1：使用PIL保存
            from PIL import Image
            import numpy as np
            pil_img = Image.fromarray(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
            pil_img.save(file_path)
            
            # 如果上面方法不行，可以尝试方法2
            # 方法2：使用OpenCV的imencode
            # ext = os.path.splitext(file_path)[1]
            # ret, buf = cv2.imencode(ext, self.processed_image)
            # if ret:
            #     with open(file_path, 'wb') as f:
            #         f.write(buf)
            
            self.status_var.set(f"已保存: {os.path.basename(file_path)}")
            messagebox.showinfo("成功", f"图片已保存至:\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存图片失败: {str(e)}")
    
    def display_image(self, cv_image, canvas, title):
        if cv_image is None:
            return
        
        # 清除画布
        canvas.delete("all")
        
        # 获取画布尺寸
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # 如果画布尚未渲染，使用更大的默认尺寸
        if canvas_width <= 1:
            canvas_width = 400
        if canvas_height <= 1:
            canvas_height = 300
            
        # 调试信息
        print(f"Canvas size: {canvas_width}x{canvas_height}")
        print(f"Image shape: {cv_image.shape}")
        
        # 转换OpenCV图像为PIL图像
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image_rgb)
        
        # 计算缩放比例以适应画布
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        # 缩放图像
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # 转换为PhotoImage并保存引用（防止垃圾回收）
        photo = ImageTk.PhotoImage(pil_image)
        canvas.image = photo  # 保存引用
        
        # 在画布中央显示图像
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)
        
        # 添加标题
        canvas.create_text(10, 10, text=title, anchor=tk.NW, fill="#333333")
    
    def show_about(self):
        messagebox.showinfo(
            "关于",
            "图片去水印工具 v1.0\n\n"
            "这是一个使用OpenCV和Tkinter开发的图片去水印工具，\n"
            "可以自动检测并去除图片中的水印，同时保持图片其他部分不变。\n\n"
            "使用方法：\n"
            "1. 点击'打开图片'选择需要处理的图片\n"
            "2. 点击'去除水印'进行处理\n"
            "3. 点击'保存结果'保存处理后的图片"
        )


def main():
    app = WatermarkRemoverApp()
    app.mainloop()

if __name__ == "__main__":
    main()