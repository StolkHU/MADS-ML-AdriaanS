# debug_dataloader.py
# Let's see exactly what's happening with the data loading

from mads_datasets import DatasetFactoryProvider, DatasetType
from torchvision import transforms

print("=== DEBUGGING DATALOADER ===")

# Test the mads_datasets step by step
print("1. Creating dataset factory...")
try:
    flowers_factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    print("✅ Factory created successfully")
except Exception as e:
    print(f"❌ Factory creation failed: {e}")
    exit()

print("\n2. Setting up transforms...")
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
print("✅ Transforms created")

print("\n3. Creating datastreamer...")
try:
    streamers = flowers_factory.create_datastreamer(
        batchsize=16,
        train_transform=train_transform,
        val_transform=val_transform,
    )
    print("✅ Datastreamer created")
    print(f"Type of streamers: {type(streamers)}")
    print(f"Streamers content: {streamers}")

    # Check what keys are available
    if isinstance(streamers, dict):
        print(f"Available keys: {list(streamers.keys())}")

        # Try to get train and val
        if "train" in streamers:
            train_loader = streamers["train"]
            print(f"✅ Train loader found: {type(train_loader)}")
        else:
            print("❌ No 'train' key found")

        if "val" in streamers:
            val_loader = streamers["val"]
            print(f"✅ Val loader found: {type(val_loader)}")
        else:
            print("❌ No 'val' key found")

        # Try different possible keys
        possible_keys = ["validation", "test", "dev"]
        for key in possible_keys:
            if key in streamers:
                print(f"Found alternative key: '{key}'")
    else:
        print(f"Streamers is not a dict: {streamers}")

except Exception as e:
    print(f"❌ Datastreamer creation failed: {e}")
    import traceback

    traceback.print_exc()

print("\n4. Testing simple data loading...")
try:
    # Try the simplest possible approach
    simple_streamers = flowers_factory.create_datastreamer(batchsize=8)
    print(f"Simple streamers: {simple_streamers}")
    print(f"Simple streamers type: {type(simple_streamers)}")

    if isinstance(simple_streamers, dict):
        print(f"Simple streamers keys: {list(simple_streamers.keys())}")

except Exception as e:
    print(f"❌ Simple datastreamer failed: {e}")
    import traceback

    traceback.print_exc()
