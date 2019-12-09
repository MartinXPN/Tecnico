import React, { Component } from 'react'
// @ts-ignore
import ReactSearchBox from 'react-search-box'

interface Props {
    countries: Array<string>;
}

export default class SearchBox extends Component<Props> {
    private readonly data: Array<{key: string, value: string}>;

    constructor(props: Props) {
        super(props);

        this.data = props.countries.map(d => {return {key: d, value: d}});
    }


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