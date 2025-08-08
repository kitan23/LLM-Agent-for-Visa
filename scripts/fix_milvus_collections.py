#!/usr/bin/env python3
"""
Script to check and fix Milvus collections with wrong dimensions
"""

from pymilvus import MilvusClient, connections
import traceback

def fix_milvus_collections():
    try:
        # Connect to Docker Milvus (not Milvus Lite)
        client = MilvusClient(uri="http://localhost:19530")
        print("✅ Connected to Docker Milvus at localhost:19530")
        
        # List all collections
        collections = client.list_collections()
        print(f"📋 Found {len(collections)} collections: {collections}")
        
        for collection_name in collections:
            try:
                # Get collection info
                collection_info = client.describe_collection(collection_name)
                print(f"\n🔍 Collection: {collection_name}")
                print(f"   Schema: {collection_info}")
                
                # Look for vector fields and their dimensions
                if 'fields' in collection_info:
                    for field in collection_info['fields']:
                        if field.get('type') == 'FloatVector':
                            current_dim = field.get('params', {}).get('dim', 'unknown')
                            print(f"   📏 Vector field '{field['name']}': {current_dim} dimensions")
                            
                            if current_dim != 384:
                                print(f"   ❌ Wrong dimension! Expected 384, got {current_dim}")
                                print(f"   🗑️  Dropping collection '{collection_name}'...")
                                client.drop_collection(collection_name)
                                print(f"   ✅ Collection '{collection_name}' dropped")
                            else:
                                print(f"   ✅ Correct dimension!")
                
            except Exception as e:
                print(f"   ❌ Error checking collection {collection_name}: {e}")
        
        # List collections after cleanup
        collections_after = client.list_collections()
        print(f"\n📋 Collections after cleanup: {collections_after}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Milvus: {e}")
        print("\nFull error:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    fix_milvus_collections() 