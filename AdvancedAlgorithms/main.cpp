#pragma GCC optimize ("O1")
#pragma GCC optimize ("O2")
#pragma GCC optimize ("O3")
#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")

#include <cstdio>
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

#define db(X) cerr << #X << ": " << X << endl
#define dba(X) cerr << #X << ": "; for(auto i: X)   cerr << i << " ";   cerr << endl
#define db2d(X) cerr << #X << ": \n"; for(auto x: X) { for( auto i : x )  cerr << i << " "; cerr << endl; }  cerr << endl

template< class T1, class T2 >
ostream& operator << (ostream& out, const pair<T1, T2>& p) {
    out << "[" << p.first << "," << p.second << "]";
    return out;
}


/// Segment tree to keep the maximum element
template <class T>
class MaxSegmentTree {
private:
    const static T INF = 1000000000;
    size_t size = 1;            /// number of the hidden nodes
    vector <T> tree;            /// values in the tree
    vector <size_t> label2id;   /// label to index mapping (labels will be the positions when dfsing in HL decomposition)
    size_t curId = 0;

    T get(size_t v, size_t l, size_t r, size_t ql, size_t qr) {
        printf( "get( %ld, %ld, %ld, %ld, %ld)", v, l, r, ql, qr );
        assert( 1 <= v && v <= tree.size() );
        printf( "YO\n" );
        fflush(stdout);
        if( l >= ql && r <= qr )    return tree[v];
        if( r < ql || l > qr )      return -INF;
        return max( get( 2 * v, l, (l+r) / 2, ql, qr ),
                    get( 2 * v + 1, (l+r) / 2 + 1, r, ql, qr ) );
    }

public:
    explicit MaxSegmentTree(size_t n) {
        assert(n >= 1);
        while( size < n )
            size *= 2;

        tree.resize(2 * size, -INF);
        label2id.resize(n, -1);
    }

    void build() {
        curId = -1;
        for( size_t v = size-1; v >= 1; --v )
            tree[v] = max( tree[2 * v], tree[2 * v + 1] );
        dba(label2id);
    }

    void add(size_t label, int val) {
        assert(0 <= curId && curId < size);
        assert(0 <= label && label < label2id.size());
        label2id[label] = curId;
        tree[size + curId] = val;
        ++curId;
    }

    T query(size_t left, size_t right) {
        left = label2id[left];
        right = label2id[right];
        assert( 0 <= left && left < size );
        assert( 0 <= right && right < size );
        assert( left <= right );
        return get(1, size, 2 * size - 1, size + left, size + right);
    }

    friend ostream& operator << (ostream& out, const MaxSegmentTree& maxSegment) {
        for( size_t v = 1; v < 2 * maxSegment.size; ++v ) {
            if( ( v & (v-1) ) == 0 )            out << endl;
            if( maxSegment.tree[v] == -INF )    out << "- ";
            else                                out << maxSegment.tree[v] << " ";
        }
        out << endl;
        return out;
    }
};

int main() {
    freopen("sample.in", "r", stdin);
    int n;
    cin >> n;
    db(n);

    int MST = 0;
    vector< vector <pair <int, int> > > g(n);    /// undirected weighted graph with n vertices
                                                 /// as it's a tree => there are n - 1 edges
    for( int i=1; i < n; ++i ) {
        int a, b, w;
        cin >> a >> b >> w;
        g[a].push_back( {b, w} );
        g[b].push_back( {a, w} );
        MST += w;
    }
    db2d(g);
    db(MST);


    MaxSegmentTree <int> maxSegments(n);
    for( int i=0; i < 8; ++i )
        maxSegments.add(7-i, i);

    maxSegments.build();
    cout << maxSegments << endl;

    cout << maxSegments.query(7, 5);


    int q;
    cin >> q;
    db(q);
    while( q-- ) {
        int a, b, w;
        cin >> a >> b >> w;

        /// TODO: query max element in the path from a to b
        int maxWeight = 4;

        if( w >= maxWeight )    cout << MST << " " << MST - maxWeight + w << endl;  /// no change in the MST
        else                    cout << MST - maxWeight + w << " " << MST << endl;  /// MST changed and became second min
    }
    return 0;
}
