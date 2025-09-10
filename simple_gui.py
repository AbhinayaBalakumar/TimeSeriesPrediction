#!/usr/bin/env python3
"""
Simple GUI for Time Series Prediction Project
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import subprocess
import os
import sys
from datetime import datetime

class SimpleTimeSeriesGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Time Series Prediction Tool")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.running = False
        self.current_demo = tk.StringVar(value="quick")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=5, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, 
                              text="🚀 AI Time Series Prediction & Forecasting",
                              font=('Arial', 18, 'bold'), 
                              fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='white', relief='raised', bd=2)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Demo selection
        demo_frame = tk.LabelFrame(main_frame, text="Choose Your Demo", 
                                  font=('Arial', 14, 'bold'), bg='white')
        demo_frame.pack(fill='x', padx=20, pady=20)
        
        # Demo options
        demos = [
            ("🎯 Quick Demo (91% improvement)", "quick", "python3 quick_demo.py"),
            ("📈 Real Stock Analysis (AAPL)", "stock", "python3 run_main_demo.py"),
            ("🔧 Multi-Dataset Analysis", "multi", "python3 run_lightweight.py"),
            ("🏆 Complete Showcase", "showcase", "python3 showcase_demo.py")
        ]
        
        self.demo_commands = {}
        
        for text, value, command in demos:
            tk.Radiobutton(demo_frame, text=text, variable=self.current_demo,
                          value=value, bg='white', font=('Arial', 12),
                          anchor='w').pack(fill='x', padx=20, pady=8)
            self.demo_commands[value] = command
        
        # Control buttons
        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.pack(fill='x', padx=20, pady=20)
        
        self.run_button = tk.Button(button_frame,
                                   text="🚀 Run Selected Demo",
                                   font=('Arial', 14, 'bold'),
                                   bg='#27ae60', fg='white',
                                   padx=30, pady=15,
                                   command=self.run_demo)
        self.run_button.pack(side='left', padx=10)
        
        tk.Button(button_frame, text="📁 Open Results",
                 font=('Arial', 12), bg='#3498db', fg='white',
                 padx=20, pady=15, command=self.open_results).pack(side='left', padx=10)
        
        tk.Button(button_frame, text="🔄 Refresh",
                 font=('Arial', 12), bg='#f39c12', fg='white',
                 padx=20, pady=15, command=self.refresh_output).pack(side='left', padx=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill='x', padx=20, pady=10)
        
        # Output area
        output_frame = tk.LabelFrame(main_frame, text="Output & Results", 
                                    font=('Arial', 12, 'bold'), bg='white')
        output_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.output_text = scrolledtext.ScrolledText(output_frame,
                                                    height=15, width=80,
                                                    font=('Courier', 10),
                                                    bg='#2c3e50', fg='#ecf0f1')
        self.output_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_bar = tk.Frame(self.root, bg='#34495e', height=25)
        self.status_bar.pack(fill='x', side='bottom')
        self.status_bar.pack_propagate(False)
        
        self.status_label = tk.Label(self.status_bar,
                                    text="Ready - Select a demo and click Run",
                                    bg='#34495e', fg='white',
                                    font=('Arial', 10))
        self.status_label.pack(side='left', padx=10, pady=2)
        
        # Initial message
        self.show_welcome_message()
        
    def show_welcome_message(self):
        """Show welcome message."""
        welcome_msg = """
🚀 Welcome to the Time Series Prediction Tool!

This AI-powered application demonstrates advanced forecasting techniques:

📊 Available Demos:
• Quick Demo: See 91% improvement in prediction accuracy (30 seconds)
• Real Stock Analysis: Analyze Apple (AAPL) stock with R² = 0.9962 (60 seconds)  
• Multi-Dataset Analysis: Test stock, climate & sales forecasting (90 seconds)
• Complete Showcase: Run all demos with comprehensive analysis (3 minutes)

🎯 How to Use:
1. Select a demo from the options above
2. Click "🚀 Run Selected Demo" 
3. Watch the progress and view results below
4. Check visualizations in the results folder

✨ Features:
• Real stock market data (Yahoo Finance)
• 80+ engineered features (technical indicators, patterns)
• Multiple ML models (Linear Regression, Random Forest)
• Professional visualizations and reporting
• Up to 91% improvement in prediction accuracy

Ready to start? Select a demo and click Run! 🎉
        """
        
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, welcome_msg)
        
    def run_demo(self):
        """Run the selected demo."""
        if self.running:
            messagebox.showwarning("Warning", "A demo is already running!")
            return
            
        selected = self.current_demo.get()
        if selected not in self.demo_commands:
            messagebox.showerror("Error", "Please select a demo to run!")
            return
            
        command = self.demo_commands[selected]
        demo_names = {
            "quick": "Quick Feature Engineering Demo",
            "stock": "Real Stock Data Analysis", 
            "multi": "Multi-Dataset Analysis",
            "showcase": "Complete Showcase"
        }
        
        demo_name = demo_names[selected]
        
        # Update UI
        self.running = True
        self.run_button.config(state='disabled', text="Running...")
        self.progress.start()
        self.status_label.config(text=f"Running {demo_name}...")
        
        # Clear output
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"🚀 Starting {demo_name}...\n\n")
        self.output_text.see(tk.END)
        
        # Run in thread
        thread = threading.Thread(target=self._run_command, args=(command, demo_name))
        thread.daemon = True
        thread.start()
        
    def _run_command(self, command, demo_name):
        """Run command in background thread."""
        try:
            # Run the command
            process = subprocess.Popen(command.split(), 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True, bufsize=1, universal_newlines=True)
            
            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.root.after(0, self._update_output, output)
            
            # Get final result
            stdout, stderr = process.communicate()
            
            # Update UI in main thread
            self.root.after(0, self._command_completed, process.returncode, stdout, stderr, demo_name)
            
        except Exception as e:
            self.root.after(0, self._command_error, str(e), demo_name)
            
    def _update_output(self, text):
        """Update output text in main thread."""
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        
    def _command_completed(self, returncode, stdout, stderr, demo_name):
        """Handle command completion."""
        self.running = False
        self.run_button.config(state='normal', text="🚀 Run Selected Demo")
        self.progress.stop()
        
        if returncode == 0:
            self.status_label.config(text=f"✅ {demo_name} completed successfully!")
            self.output_text.insert(tk.END, f"\n✅ {demo_name} completed successfully!\n")
            self.output_text.insert(tk.END, f"📊 Check the results folder for visualizations.\n")
            
            if stdout:
                self.output_text.insert(tk.END, f"\n{stdout}")
                
            messagebox.showinfo("Success", 
                              f"{demo_name} completed successfully!\n\n"
                              f"Check the output below and click 'Open Results' "
                              f"to view visualizations.")
        else:
            self.status_label.config(text=f"❌ {demo_name} failed")
            self.output_text.insert(tk.END, f"\n❌ {demo_name} failed!\n")
            if stderr:
                self.output_text.insert(tk.END, f"Error: {stderr}\n")
            messagebox.showerror("Error", f"{demo_name} failed:\n\n{stderr}")
            
        self.output_text.see(tk.END)
        
    def _command_error(self, error, demo_name):
        """Handle command error."""
        self.running = False
        self.run_button.config(state='normal', text="🚀 Run Selected Demo")
        self.progress.stop()
        self.status_label.config(text=f"❌ {demo_name} error")
        self.output_text.insert(tk.END, f"\n❌ Error running {demo_name}: {error}\n")
        messagebox.showerror("Error", f"Error running {demo_name}:\n\n{error}")
        
    def open_results(self):
        """Open the results folder."""
        results_path = os.path.abspath("results")
        if os.path.exists(results_path):
            try:
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", results_path])
                elif sys.platform == "win32":  # Windows
                    subprocess.run(["explorer", results_path])
                else:  # Linux
                    subprocess.run(["xdg-open", results_path])
                self.status_label.config(text="📁 Opened results folder")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open results folder:\n{e}")
        else:
            messagebox.showinfo("Info", "Results folder not found. Run a demo first!")
            
    def refresh_output(self):
        """Refresh the output display."""
        if os.path.exists("results"):
            files = [f for f in os.listdir("results") if f.endswith(('.png', '.log', '.json'))]
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "📊 Results Directory Contents:\n\n")
            
            for file in sorted(files):
                if file.endswith('.png'):
                    self.output_text.insert(tk.END, f"📈 {file}\n")
                elif file.endswith('.log'):
                    self.output_text.insert(tk.END, f"📝 {file}\n")
                elif file.endswith('.json'):
                    self.output_text.insert(tk.END, f"📋 {file}\n")
                    
            self.output_text.insert(tk.END, f"\nTotal files: {len(files)}\n")
            self.output_text.insert(tk.END, f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.status_label.config(text="🔄 Results refreshed")
        else:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "No results found. Run a demo first!")

def main():
    """Main function."""
    root = tk.Tk()
    
    # Set icon and center window
    try:
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
    except:
        pass
    
    app = SimpleTimeSeriesGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()