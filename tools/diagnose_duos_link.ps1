param(
  [string]$BoardIp = "192.168.42.1",
  [string]$SshHost = "milkv-duo",
  [int]$SshPort = 22,
  [switch]$SkipSsh
)

$ErrorActionPreference = "Continue"

function Write-Section {
  param([string]$Title)
  Write-Host ""
  Write-Host "== $Title =="
}

function Write-Check {
  param(
    [string]$Name,
    [bool]$Ok,
    [string]$Detail
  )
  $status = if ($Ok) { "OK" } else { "FAIL" }
  Write-Host ("[{0}] {1}: {2}" -f $status, $Name, $Detail)
}

function Test-TcpPort {
  param(
    [string]$ComputerName,
    [int]$Port,
    [int]$TimeoutMs = 3000
  )
  $client = New-Object System.Net.Sockets.TcpClient
  try {
    $async = $client.BeginConnect($ComputerName, $Port, $null, $null)
    $ok = $async.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
    if (-not $ok) {
      return $false
    }
    $client.EndConnect($async)
    return $true
  } catch {
    return $false
  } finally {
    $client.Close()
  }
}

$failures = 0

Write-Section "Windows network adapters"
$adapters = Get-NetAdapter | Sort-Object Name
$adapters | Select-Object Name, InterfaceDescription, Status, LinkSpeed, MacAddress | Format-Table -AutoSize

$usbNetAdapters = $adapters | Where-Object {
  $_.Name -match "USB|RNDIS|NCM|CDC|GAIA|Duo|Milk" -or
  $_.InterfaceDescription -match "USB|RNDIS|NCM|CDC|GAIA|Duo|Milk|Ethernet Gadget"
}
if ($usbNetAdapters.Count -gt 0) {
  Write-Check "USB network adapter" $true ("found {0} candidate adapter(s)" -f $usbNetAdapters.Count)
  $usbNetAdapters | Select-Object Name, InterfaceDescription, Status, LinkSpeed, MacAddress | Format-Table -AutoSize
} else {
  Write-Check "USB network adapter" $false "no CDC-NCM/RNDIS/GAIA-like adapter is visible"
  $failures++
}

Write-Section "Relevant PnP devices"
try {
  $pnp = Get-PnpDevice -PresentOnly | Where-Object {
    $_.FriendlyName -match "GAIA|RNDIS|NCM|CDC|USB Serial|COM|Duo|Milk|Ethernet Gadget" -or
    $_.InstanceId -match "VID_|USB"
  } | Sort-Object Class, FriendlyName
  if ($pnp.Count -gt 0) {
    $pnp | Select-Object Status, Class, FriendlyName, InstanceId | Format-Table -AutoSize
  } else {
    Write-Host "No relevant present PnP devices found."
  }
} catch {
  Write-Host "Get-PnpDevice unavailable or failed: $($_.Exception.Message)"
}

Write-Section "IP reachability"
$pingOk = Test-Connection -ComputerName $BoardIp -Count 1 -Quiet -ErrorAction SilentlyContinue
Write-Check "ping $BoardIp" $pingOk $(if ($pingOk) { "board replied" } else { "no ICMP reply" })
if (-not $pingOk) { $failures++ }

$tcpOk = Test-TcpPort -ComputerName $BoardIp -Port $SshPort -TimeoutMs 3000
Write-Check "tcp ${BoardIp}:$SshPort" $tcpOk $(if ($tcpOk) { "SSH port reachable" } else { "SSH port not reachable" })
if (-not $tcpOk) { $failures++ }

Write-Section "SSH"
if ($SkipSsh) {
  Write-Host "Skipped by --SkipSsh."
} else {
  $sshCmd = Get-Command ssh -ErrorAction SilentlyContinue
  if ($null -eq $sshCmd) {
    Write-Check "ssh command" $false "OpenSSH client not found in PATH"
    $failures++
  } else {
    $sshOutput = & cmd /c "ssh -o BatchMode=yes -o ConnectTimeout=5 $SshHost ""uname -a; ip addr show usb0 2>/dev/null || ip addr"" 2>&1"
    $sshOutput
    $sshOk = ($LASTEXITCODE -eq 0)
    Write-Check "ssh $SshHost" $sshOk $(if ($sshOk) { "login works" } else { "login failed or timed out" })
    if (-not $sshOk) { $failures++ }
  }
}

Write-Section "Serial ports"
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $pythonCmd) {
  Write-Check "python serial.tools.list_ports" $false "python not found in PATH"
  $failures++
} else {
  $ports = & cmd /c "python -m serial.tools.list_ports 2>&1"
  $portText = ($ports | Out-String).Trim()
  if ([string]::IsNullOrWhiteSpace($portText)) {
    $portText = "no output"
  }
  Write-Host $portText
  $hasPort = ($portText -notmatch "^no ports found$")
  Write-Check "serial port enumeration" $hasPort $(if ($hasPort) { "at least one serial port is visible" } else { "no serial ports found" })
  if (-not $hasPort) { $failures++ }
}

Write-Section "Summary"
if ($failures -eq 0) {
  Write-Host "DuoS host link looks ready for deploy_live_runner.sh."
  exit 0
}

Write-Host ("{0} check(s) failed. Fix USB driver/cable/power or board boot before deploying runtime code." -f $failures)
exit 1
