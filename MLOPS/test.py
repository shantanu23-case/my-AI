from zenml.client import Client

client = Client()
client.activate_stack("temp_stack")

print("✅ Global active stack set to:", client.active_stack.name)
