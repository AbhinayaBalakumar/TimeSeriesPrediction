#!/usr/bin/env python3
"""
GUI Launcher Menu for Time Series Prediction Project
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

class GUILauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Time Series Prediction - GUI Launcher")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the launcher UI."""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, 
                              text="🚀 Time Series Prediction Tool",
                              font=('Arial', 20, 'bold'), 
                              fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(title_frame,
                                 text="Choose Your Interface",
                                 font=('Arial', 12), 
                                 fg='#ecf0f1', bg='#2c3e50')
        subtitle_label.pack()
        
        # Main content
        main_frame = tk.Frame(self.root, bg='white', relief='raised', bd=2)
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Options
        options_frame = tk.Frame(main_frame, bg='white')
        options_frame.pack(fill='both', expand=True, padx=30, pady=30)
        
        # GUI Options
        self.create_option_button(options_frame, 
                                 "🖥️ Desktop GUI Application",
                                 "Full-featured desktop interface with real-time output",
                                 self.launch_desktop_gui, 0)
        
        self.create_option_button(options_frame,
                                 "🌐 Web Browser Interface", 
                                 "Modern web-based interface accessible from any browser",
                                 self.launch_web_gui, 1)
        
        self.create_option_button(options_frame,
                                 "⚡ Quick Command Line Demo",
                                 "Run the fastest demo directly (91% improvement)",
                                 self.run_quick_demo, 2)
        
        self.create_option_button(options_frame,
                                 "📊 Real Stock Analysis",
                                 "Analyze Apple stock data directly (R² = 0.9962)",
                                 self.run_stock_demo, 3)
        
        # Info section
        info_frame = tk.Frame(main_frame, bg='#ecf0f1', relief='sunken', bd=1)
        info_frame.pack(fill='x', padx=20, pady=10)
        
        info_text = """
💡 Quick Info:
• Desktop GUI: Best for interactive use with real-time feedback
• Web Interface: Great for remote access and modern UI experience  
• Quick Demo: See 91% improvement in 30 seconds
• Stock Analysis: Real Apple data with professional results

All interfaces provide the same powerful AI forecasting capabilities!
        """
        
        info_label = tk.Label(info_frame, text=info_text,
                             font=('Arial', 10), bg='#ecf0f1',
                             justify='left', anchor='w')
        info_label.pack(padx=15, pady=10)
        
        # Footer
        footer_frame = tk.Frame(self.root, bg='#34495e', height=30)
        footer_frame.pack(fill='x', side='bottom')
        footer_frame.pack_propagate(False)
        
        footer_label = tk.Label(footer_frame,
                               text="Ready to explore AI-powered time series forecasting!",
                               bg='#34495e', fg='white',
                               font=('Arial', 10))
        footer_label.pack(expand=True)
        
    def create_option_button(self, parent, title, description, command, row):
        """Create an option button."""
        button_frame = tk.Frame(parent, bg='#ecf0f1', relief='raised', bd=2)
        button_frame.pack(fill='x', pady=8)
        
        # Title button
        title_btn = tk.Button(button_frame,
                             text=title,
                             font=('Arial', 12, 'bold'),
                             bg='#3498db', fg='white',
                             padx=20, pady=10,
                             command=command,
                             cursor='hand2')
        title_btn.pack(fill='x', padx=10, pady=(10, 5))
        
        # Description
        desc_label = tk.Label(button_frame,
                             text=description,
                             font=('Arial', 10),
                             bg='#ecf0f1', fg='#2c3e50')
        desc_label.pack(padx=10, pady=(0, 10))
        
    def launch_desktop_gui(self):
        """Launch the desktop GUI."""
        try:
            subprocess.Popen([sys.executable, 'simple_gui.py'])
            messagebox.showinfo("Success", "Desktop GUI launched successfully!")
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch desktop GUI:\n{e}")
            
    def launch_web_gui(self):
        """Launch the web GUI."""
        try:
            subprocess.Popen([sys.executable, 'web_gui.py'])
            messagebox.showinfo("Success", 
                              "Web server starting...\n\n"
                              "Open your browser and go to:\n"
                              "http://localhost:5000")
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch web GUI:\n{e}")
            
    def run_quick_demo(self):
        """Run the quick demo directly."""
        try:
            subprocess.Popen([sys.executable, 'quick_demo.py'])
            messagebox.showinfo("Success", 
                              "Quick demo started!\n\n"
                              "Check the terminal for output and\n"
                              "results/ folder for visualizations.")
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run quick demo:\n{e}")
            
    def run_stock_demo(self):
        """Run the stock analysis demo directly."""
        try:
            subprocess.Popen([sys.executable, 'run_main_demo.py'])
            messagebox.showinfo("Success", 
                              "Stock analysis started!\n\n"
                              "Check the terminal for output and\n"
                              "results/ folder for visualizations.")
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run stock demo:\n{e}")

def main():
    """Main function."""
    root = tk.Tk()
    
    # Center window
    try:
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
    except:
        pass
    
    app = GUILauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main()