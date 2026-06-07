# ACI Assignment 1 - PS11: Route Optimizer

**Local Beam Search (k=2) Implementation**

**Course**: MTech in Artificial Intelligence and Machine Learning
**Assignment**: Route Optimizer Agent using Local Beam Search
**Group**: G091

---

## Team/Contributors

| Name | Email | ID | Contribution |
|------|-------|----|-------------|
| Darshan L S | 2025aa05828@wilp.bits-pilani.ac.in | 2025aa05828 | 100% |
| RANJITA PATEL | 2025ab05117@wilp.bits-pilani.ac.in | 2025ab05117 | 100% |
| SIHAAM S | 2025aa05354@wilp.bits-pilani.ac.in | 2025aa05354 | 100% |
| SPOORTHY N KUMAR | 2025ab05027@wilp.bits-pilani.ac.in | 2025ab05027 | 100% |
| MAHADEVA SWAMY B N | 2025ab05081@wilp.bits-pilani.ac.in | 2025ab05081 | 100% |

---

## Problem Description

An intelligent agent must find the **optimal low-cost route** from Start (`S`) to Goal (`G`) in a 5x5 warehouse grid.  
- Moves allowed: **Up, Down, Left, Right** (No diagonals)  
- Normal cells cost = **1**  
- High-cost cells (`C`) cost = **3** (1 + 2 penalty)  
- Blocked cells (`X`) cannot be traversed  
- Heuristic used: **Manhattan Distance**
