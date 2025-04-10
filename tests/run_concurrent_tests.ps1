# PowerShell script to run concurrent access tests
$ErrorActionPreference = "Stop"

# Configuration
$baseThreadCount = 5
$baseOperationsPerThread = 1000
$outputFile = "concurrent_results.csv"
$pythonCmd = "python"

# Create CSV header (the Python script will create this if needed)
# We're just making sure the file exists
if (-not (Test-Path $outputFile)) {
    "" | Out-File $outputFile
}

# Test with increasing thread counts (simulating more concurrent users/clients)
Write-Host "Running concurrent access tests with increasing thread counts..."
foreach ($scale in 1..5) {
    $threadCount = $baseThreadCount * $scale
    $operations = $baseOperationsPerThread
    Write-Host "Testing with $threadCount concurrent threads..."
    
    try {
        # Set PYTHONPATH and run the test
        $env:PYTHONPATH = "..;$env:PYTHONPATH"
        & $pythonCmd "concurrent_test.py" --threads $threadCount --operations $operations --output $outputFile
        Write-Host "Completed test with $threadCount threads"
    }
    catch {
        Write-Error "Error running test with $threadCount threads: $($_.Exception.Message)"
    }
    
    # Short pause between tests
    Start-Sleep -Seconds 5
}

# Test with different write/read ratios
Write-Host "`nRunning concurrent access tests with different write/read ratios..."
$threadCount = $baseThreadCount * 2  # Use a moderate thread count
$writeRatios = @(0.25, 0.5, 0.75, 0.9)

foreach ($ratio in $writeRatios) {
    Write-Host "Testing with write ratio $ratio..."
    
    try {
        # Set PYTHONPATH and run the test
        $env:PYTHONPATH = "..;$env:PYTHONPATH"
        & $pythonCmd "concurrent_test.py" --threads $threadCount --operations $operations --write-ratio $ratio --output $outputFile
        Write-Host "Completed test with write ratio $ratio"
    }
    catch {
        Write-Error "Error running test with write ratio $ratio: $($_.Exception.Message)"
    }
    
    # Short pause between tests
    Start-Sleep -Seconds 5
}

# Test with high load
Write-Host "`nRunning high load concurrent test..."
$highThreadCount = $baseThreadCount * 10
$highOperations = $baseOperationsPerThread / 2  # Reduce operations to avoid excessive runtime

try {
    # Set PYTHONPATH and run the test
    $env:PYTHONPATH = "..;$env:PYTHONPATH"
    & $pythonCmd "concurrent_test.py" --threads $highThreadCount --operations $highOperations --output $outputFile
    Write-Host "Completed high load test"
}
catch {
    Write-Error "Error running high load test: $($_.Exception.Message)"
}

# Make sure Redis is running before running race condition test
Write-Host "`nRunning race condition test (many threads, few operations)..."
$raceThreadCount = 100
$raceOperations = 50  # Each thread does fewer operations but with high concurrency

try {
    # Set PYTHONPATH and run the test
    $env:PYTHONPATH = "..;$env:PYTHONPATH"
    & $pythonCmd "concurrent_test.py" --threads $raceThreadCount --operations $raceOperations --output $outputFile
    Write-Host "Completed race condition test"
}
catch {
    Write-Error "Error running race condition test: $($_.Exception.Message)"
}

# Generate summary report
Write-Host "`nAll concurrent tests completed. Results saved to $outputFile"

# Optional: You can add code here to parse the CSV and generate a summary report
if (Test-Path $outputFile) {
    Write-Host "`nTest Results Summary:"
    $results = Import-Csv $outputFile
    
    # Calculate averages
    $avgThroughput = ($results | Measure-Object -Property "Throughput" -Average).Average
    $avgLatency = ($results | Measure-Object -Property "AvgLatency" -Average).Average
    $avgP95Latency = ($results | Measure-Object -Property "P95Latency" -Average).Average
    
    Write-Host "Average throughput across all tests: $([math]::Round($avgThroughput, 2)) ops/sec"
    Write-Host "Average latency across all tests: $([math]::Round($avgLatency, 2)) ms"
    Write-Host "Average P95 latency across all tests: $([math]::Round($avgP95Latency, 2)) ms"
    
    # Find best and worst cases
    $bestThroughput = ($results | Sort-Object -Property "Throughput" -Descending | Select-Object -First 1)
    $worstLatency = ($results | Sort-Object -Property "P95Latency" -Descending | Select-Object -First 1)
    
    Write-Host "`nBest throughput: $($bestThroughput.Throughput) ops/sec with $($bestThroughput.Threads) threads and $($bestThroughput.WriteRatio) write ratio"
    Write-Host "Worst P95 latency: $($worstLatency.P95Latency) ms with $($worstLatency.Threads) threads and $($worstLatency.WriteRatio) write ratio"
} 