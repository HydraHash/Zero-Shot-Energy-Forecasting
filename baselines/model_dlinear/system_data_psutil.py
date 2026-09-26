import os
import time
import psutil
import threading
from pathlib import Path

class ResourceProfiler:
    """
    A universal context manager to track execution time, CPU, and RAM usage.
    Outputs a standardized text report for model comparison.
    """
    def __init__(self, run_name, input_file_path, output_dir_path):
        self.run_name = run_name
        self.input_file_path = Path(input_file_path)
        self.output_dir_path = Path(output_dir_path)
        
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.cpu_measurements = []
        self.ram_measurements = []
        
    def _monitor_resources(self):
        # Sample CPU and RAM every 0.1 seconds in the background
        while not self.stop_event.is_set():
            self.cpu_measurements.append(self.process.cpu_percent(interval=None))
            # rss is Resident Set Size (Physical RAM used by this process)
            self.ram_measurements.append(self.process.memory_info().rss / (1024 * 1024)) # Convert to MB
            time.sleep(0.1)

    def __enter__(self):
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        
        # Prime the CPU percent calculation
        self.process.cpu_percent(interval=None) 
        
        # Start background monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.monitor_thread.join()
        self.end_time = time.time()
        
        self._generate_report()

    def _generate_report(self):
        execution_time = self.end_time - self.start_time
        avg_cpu = sum(self.cpu_measurements) / len(self.cpu_measurements) if self.cpu_measurements else 0
        peak_ram = max(self.ram_measurements) if self.ram_measurements else 0
        
        report_lines = [
            "==================================================",
            f" PERFORMANCE REPORT: {self.run_name}",
            "==================================================",
            f"Date Recorded : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "[I/O Paths]",
            f"Input File    : {self.input_file_path.name}",
            f"Input Dir     : {self.input_file_path.parent.absolute()}",
            f"Output Dir    : {self.output_dir_path.absolute()}",
            "",
            "[Resource Consumption]",
            f"Wall-Clock Time : {execution_time:.2f} seconds",
            f"Peak RAM Usage  : {peak_ram:.2f} MB",
            f"Average CPU Load: {avg_cpu:.2f} %",
            "=================================================="
        ]
        
        report_path = self.output_dir_path / f"performance_report_{self.run_name}.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        
        print(f" -> Profiling report saved: {report_path.name}")