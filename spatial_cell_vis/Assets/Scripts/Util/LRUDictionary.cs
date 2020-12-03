using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

// Ref: https://stackoverflow.com/a/3719378
namespace LRUDictionary
{
    public class LRUDictionary<K, V>
    {
        private int capacity;
        private Dictionary<K, LinkedListNode<LRUDictionaryItem<K, V>>> cacheMap = new Dictionary<K, LinkedListNode<LRUDictionaryItem<K, V>>>();
        private LinkedList<LRUDictionaryItem<K, V>> lruList = new LinkedList<LRUDictionaryItem<K, V>>();
        private Action<V> disposer;

        public LRUDictionary(int capacity, Action<V> disposer = null)
        {
            this.capacity = capacity;
            this.disposer = disposer;
        }

        [MethodImpl(MethodImplOptions.Synchronized)]
        public V get(K key)
        {
            LinkedListNode<LRUDictionaryItem<K, V>> node;
            if (cacheMap.TryGetValue(key, out node))
            {
                V value = node.Value.value;
                //lruList.Remove(node);
                //lruList.AddLast(node);
                return value;
            }
            return default(V);
        }

        [MethodImpl(MethodImplOptions.Synchronized)]
        public void add(K key, V val)
        {
            if (cacheMap.Count >= capacity)
            {
                RemoveFirst();
            }

            LRUDictionaryItem<K, V> cacheItem = new LRUDictionaryItem<K, V>(key, val);
            LinkedListNode<LRUDictionaryItem<K, V>> node = new LinkedListNode<LRUDictionaryItem<K, V>>(cacheItem);
            lruList.AddLast(node);
            cacheMap.Add(key, node);
        }

        private void RemoveFirst()
        {
            // Remove from LRUPriority
            LinkedListNode<LRUDictionaryItem<K, V>> node = lruList.First;
            lruList.RemoveFirst();

            // Remove from cache
            cacheMap.Remove(node.Value.key);

            disposer?.Invoke(node.Value.value);
        }

        [MethodImpl(MethodImplOptions.Synchronized)]
        public IEnumerator<LRUDictionaryItem<K, V>> GetEnumerator()
        {
            return lruList.GetEnumerator();
        }
    }

    public class LRUDictionaryItem<K, V>
    {
        public LRUDictionaryItem(K k, V v)
        {
            key = k;
            value = v;
        }
        public K key;
        public V value;
    }
}
