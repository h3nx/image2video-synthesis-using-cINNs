
iterations=10
echo "Starting coldstart test for lightning generation with "
for ((i = 0 ; i < iterations ; i++)); do
  echo $i
  D:\\BTH\\EXJOBB\\ColabServers\\Image2video\\venv\\Scripts\\python.exe D:/BTH/EXJOBB/ColabServers/Image2video/PerformanceTestCold.py
done

echo "coldstart test done"
sleep 5

