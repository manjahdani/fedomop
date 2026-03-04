# Troubleshooting
## Permission error

Cause:
Incorrect permissions on `flwr_apps`.

Fix:
```bash
sudo chown -R $USER:$USER ~/flwr_apps
```

## Connection issues

Test connectivity:
```bash
nc -vz -w 3 <SUPERLINK_IP> 9092
```

Connection to <SERVER_IP> 9092 port [tcp/*] succeeded!



It should output something like this : 

If this fails:

Firewall

Port already in use

Wrong IP