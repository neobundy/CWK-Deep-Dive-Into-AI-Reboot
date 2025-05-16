# Hardware Snapshot 

*(Personal lab notebook — read if it helps; ignore if it doesn't. 🙂 Last verified 2025‑04‑22)*

## Participating & Candidate Hardware

1. Primary Workstations:

   - Office Environment:
     * Device A: Mac Studio
     * Processor: M3 Ultra
     * Memory: 512GB RAM

     * Device B: Mac Studio
     * Processor: M3 Ultra
     * Memory: 512GB RAM

     * Device C: Mac Studio
     * Processor: M2 Ultra
     * Memory: 192GB RAM
   
   - Music Production Environment:
     * Device: Mac Studio
     * Processor: M2 Ultra
     * Memory: 192GB RAM

2. Mobile Workstations: (Standalone, Candidate for future expansion)
   - Primary Laptop:
     * Device: MacBook Pro
     * Processor: M4 Max
     * Memory: 128GB RAM
   
   - Secondary Laptop:
     * Device: MacBook Pro
     * Processor: M3 Max
     * Memory: 128GB RAM
   
   - Ultraportable:
     * Device: MacBook Air
     * Processor: M3
     * Memory: 16GB RAM

3. Dedicated Small AI Computing: (Standalone, Candidate for future expansion)
   - AI Development Server:
     * Device: Mac Mini Pro
     * Processor: M4
     * Memory: 64GB RAM

4. Windows Workstation: (Standalone, Candidate for future expansion)

   - AMD Ryzen 9 7950X3D 16-core 4.20GHz
   - 128GB of RAM
   - RTX4090/24GB of RAM
   - 2TB NVMe SSD x 4
   - 10Gbps Ethernet
   - Automatic1111 Stable Diffusion Machine
   - python 3.10.6
   - CUDA 12.3

## IDs and Canonical Hostnames

Format: `cwk-<role>-<chip>-<seq>`

Where:
- `role`: h (head), w (worker), l (laptop)
- `chip`: m3u (M3 Ultra), m2u (M2 Ultra), m4x (M4 Max), m3x (M3 Max)
- `seq`: Two-digit sequence number (01, 02, etc.)

## Device Mapping

Canonical names are for easy human reference. IDs might be used in configuration files and scripts.

Any can be used in `/etc/hosts` or in other DNS‑based systems.

| ID | Canonical Name | Alias | Role | Chip | Location |
|----------------|----------------|-------|------|------|----------|
| cwk-h-m3u-01 | CWKServerM3U | head-primary | Primary Head | M3 Ultra | Office |
| cwk-h-m3u-02 | CWKOfficeM3U| head-secondary | Secondary Head | M3 Ultra | Office |
| cwk-w-m2u-03 | CWKOfficeM2U | worker-primary | Primary Worker | M2 Ultra | Office |
| cwk-w-m2u-04 | CWKMusicStudioM2U | worker-secondary | Secondary Worker | M2 Ultra | Music Studio |
| cwk-l-m4x-01 | macbookpro2024 | laptop-primary | Primary Laptop | M4 Max | Office |
| cwk-l-m3x-01 | macbookpro2023 | laptop-secondary | Secondary Laptop | M3 Max | Living Room |

## Implementation Notes

1. Both IDs and canonical names are used in:
   - System configuration files
   - Ray cluster YAML
   - `/etc/hosts` entries
   - Scripts and automation

2. Aliases are used for:
   - SSH connections
   - Bonjour Computer Names
   - Human-readable references
   - Monitoring dashboards

3. Location information is for physical placement reference only and not part of the naming scheme.

---

## Sample /etc/hosts entries

```plaintext

# 10G Network

10.0.1.x CWKMusicStudioM2U ai-server-studio cwk-w-m2u-04 worker-secondary
10.0.1.x ai-server-mini
10.0.1.x main-switch
10.0.1.x CWKServerM3U cwk-h-m3u-01 head-primary 
10.0.1.x CWKOfficeM3U cwk-h-m3u-02 head-secondary
10.0.1.x CWKOfficeM2U cwk-w-m2u-03 worker-primary

# Thunderbolt Network
# Add '.tb' in Advanced Network Settings -> Search Domain: '.tb'

192.168.1.x CWKOfficeM3U.tb cwk-h-m3u-02.tb head-secondary.tb
192.168.1.x CWKServerM3U.tb cwk-h-m3u-01.tb head-primary.tb
192.168.1.x CWKOfficeM2U.tb cwk-w-m2u-03.tb worker-primary.tb

127.0.0.1	localhost
255.255.255.255	broadcasthost
::1             localhost
```

---

[⇧ Back&nbsp;to&nbsp;README](../README.md)