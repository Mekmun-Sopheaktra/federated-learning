vagrant up

vagrant status

vagrant ssh fl-server -c "sudo bash /vagrant/setup.sh"

vagrant ssh fl-client1 -c "sudo bash /vagrant/setup.sh"

vagrant ssh fl-client2 -c "sudo bash /vagrant/setup.sh"

vagrant ssh fl-server

ping -c 3 192.168.56.11
ping -c 3 192.168.56.12

exit

# Terminal 1 Start the FL server
vagrant ssh fl-server -c "cd /vagrant/src && sudo /opt/fl-venv/bin/python server.py"

# Terminal 2 Start Client 1
vagrant ssh fl-client1 -c "cd /vagrant/src && sudo /opt/fl-venv/bin/python client.py 1"

# Terminal 3 Start Client 2
vagrant ssh fl-client2 -c "cd /vagrant/src && sudo /opt/fl-venv/bin/python client.py 2"


# Windows
pip install pandas matplotlib
python analyze.py

