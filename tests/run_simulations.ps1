# PowerShell script to run all simulations
$ErrorActionPreference = "Stop"

# Configuration
$pythonCmd = "python"
$testsDir = $PSScriptRoot
$resultsDir = Join-Path $testsDir "results"

# Create results directory if it doesn't exist
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

# Function to check if Redis is running in Docker
function Test-RedisConnection {
    try {
        $result = docker exec $(docker ps -q --filter "ancestor=redis") redis-cli ping
        return $result -eq "PONG"
    }
    catch {
        Write-Host "Error checking Redis connection: $_"
        return $false
    }
}

# Ensure Redis is running
Write-Host "Checking Redis connection..."
if (-not (Test-RedisConnection)) {
    Write-Error "Redis is not running in Docker. Please ensure the Redis container is running."
    exit 1
}
Write-Host "Redis connection confirmed."

# Run scaling tests
Write-Host "`nStarting scaling tests..."
try {
    $scalingOutput = Join-Path $resultsDir "scaling_results.csv"
    & $pythonCmd (Join-Path $testsDir "scaling_test.py") --agents 10 --memories 1000 --output $scalingOutput
    Write-Host "Scaling tests completed. Results saved to $scalingOutput"
}
catch {
    Write-Error "Error running scaling tests: $_"
}

# Short pause between test suites
Start-Sleep -Seconds 5

# Run conversation simulation
Write-Host "`nStarting conversation simulation..."
try {
    & $pythonCmd (Join-Path $testsDir "conversation_simulation.py")
    Write-Host "Conversation simulation completed"
}
catch {
    Write-Error "Error running conversation simulation: $_"
}

Write-Host "`nAll simulations completed!"
Write-Host "Results can be found in: $resultsDir"

# Optional: Generate summary report
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$summaryFile = Join-Path $resultsDir "summary_${timestamp}.txt"

Write-Host "`nGenerating summary report..."
@"
Simulation Summary Report
Generated: $(Get-Date)

1. Scaling Test Results
----------------------
$(Get-Content $scalingOutput | Out-String)

2. Conversation Simulation Results
--------------------------------
See conversation_ltm.db for detailed results
"@ | Out-File $summaryFile

Write-Host "Summary report saved to: $summaryFile" 