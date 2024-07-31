# Example structural attacks in a real network

## Infrastructure

## Instructions to reproduce the attacks

1. Download and install Marionnet (https://www.marionnet.org/site/index.php/en/), Argus (https://openargus.org/using-argus).
2. Download `test.mar` from the following [link](https://drive.google.com/file/d/13zcN91ALcBQkuAuNjekHzJA9VRs33fww/view?usp=sharing) and open the `test.mar` file inside Marionnet.
3. Start all the machines.
4. Once the hosts are up and running, run the following `tcpdump -w /mnt/hostfs/data.pcap -i any` command in the `router` host. This will put `tcpdump` in listen mode and will collect all the packets in `data.pcap`.
5. According to the attack, follow one of the following substeps:
   1. From `host1` (Compromised node) run `ping -c 1 192.168.2.1` to contact `host2` (IP Address: 192.168.2.1). This will result in a `ICMP` packet from `host1` to `host2` (C2x attack).
   2. From `host1` (Compromised node) run `hping3 -a 192.168.2.1 -c 1 192.168.3.1` to contact `host3` (IP Address 192.168.3.1) with spoofed source IP address of `host2`. This will result in a `ICMP` packet from `host1` to `host2` (C2x attack).
6. On the host machine (i.e., the one in which marionnet is installed), go to `<marionnet_project_dir>/test/hostfs/4/` and verify that the `data.pcap` file is there.
7. Execute `argus -r data.pcap -F argus.conf -w out.argus`
8. Execute `ra -r out.argus -F ra.conf > out.csv`
9. Open the resulting `out.csv` file and check that the produced netflows have the desired properties (e.g., correct source / destination IP addresses).

