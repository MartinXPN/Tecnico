import React, { Component } from 'react'
// @ts-ignore
import ReactSearchBox from 'react-search-box'

export default class SearchBox extends Component {
    data = [
        {
            key: 'john',
            value: 'John Doe',
        },
        {
            key: 'jane',
            value: 'Jane Doe',
        },
        {
            key: 'mary',
            value: 'Mary Phillips',
        },
        {
            key: 'robert',
            value: 'Robert',
        },
        {
            key: 'karius',
            value: 'Karius',
        },
    ];

    render() {
        return (
            <ReactSearchBox
                placeholder="Search for a country..."
                data={this.data}
                // @ts-ignore
                callback={record => console.log(record)}
            />
        )
    }
}