# PowerShell script to run scaling tests
$ErrorActionPreference = "Stop"

# Configuration
$baseAgentCount = 10
$baseMemoriesPerAgent = 1000
$outputFile = "scaling_results.csv"
$pythonCmd = "python"

# Create CSV header
"AgentCount,MemoriesPerAgent,STM_Latency_ms,IM_Latency_ms,LTM_Latency_ms,TotalMemories" | Out-File $outputFile

# Test with increasing agent counts
Write-Host "Running agent scaling tests..."
foreach ($scale in 1..5) {
    $agentCount = $baseAgentCount * [Math]::Pow(2, $scale - 1)
    Write-Host "Testing with $agentCount agents..."
    
    try {
        & $pythonCmd "scaling_test.py" --agents $agentCount --memories $baseMemoriesPerAgent --output $outputFile
        Write-Host "Completed test with $agentCount agents"
    }
    catch {
        Write-Error "Error running test with $agentCount agents: $_"
    }
    
    # Short pause between tests
    Start-Sleep -Seconds 5
}

# Test with increasing memories per agent
Write-Host "`nRunning memory volume scaling tests..."
foreach ($scale in 1..5) {
    $memoriesCount = $baseMemoriesPerAgent * [Math]::Pow(2, $scale - 1)
    Write-Host "Testing with $memoriesCount memories per agent..."
    
    try {
        & $pythonCmd "scaling_test.py" --agents $baseAgentCount --memories $memoriesCount --output $outputFile
        Write-Host "Completed test with $memoriesCount memories per agent"
    }
    catch {
        Write-Error "Error running test with $memoriesCount memories: $_"
    }
    
    # Short pause between tests
    Start-Sleep -Seconds 5
}

Write-Host "`nAll scaling tests completed. Results saved to $outputFile" 