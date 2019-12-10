import React, {Component} from 'react'
import Fuse from 'fuse.js';
import "./SearchBox.css";
import OutsideAlerter from './OutsideAlerter';


interface Props {
    selectedCountries: Set<string>;
    countries: Array<string>;
    addCountry: (country: string) => void;
    removeCountry: (country: string) => void;
}

interface State {
    searchResults: Array<string>;
}

export default class SearchBox extends Component<Props, State> {
    state = {
        searchResults: [],
    };
    private fuse: Fuse<{ key: string }, Fuse.FuseOptions<{ key: string }>>;

    constructor(props: Props) {
        super(props);
        const data = props.countries.map(d => {
            return {key: d}
        });

        const options: Fuse.FuseOptions<{ key: string }> = {keys: ['key']};
        this.fuse = new Fuse(data, options);
    }

    handleChange = (query: string) => {
        // @ts-ignore
        const results = this.fuse.search(query).map(record => record.key);
        this.setState({searchResults: results});
        console.log(query);
    };

    searchResultClicked = (country: string) => {
        console.log(`clicked ${country}`);
        if (this.props.selectedCountries.has(country)) {
            this.props.removeCountry(country);
        } else {
            this.props.addCountry(country);
        }
    };


    render() {
        return (
            <OutsideAlerter onOutsideClicked={() => this.handleChange("")}>
                <div className="container">
                    <input className="input"
                           placeholder="Search for countries..."
                           onChange={(event) => this.handleChange(event.target.value)}
                           onFocus={(event) => this.handleChange(event.target.value)}/>
                    {this.state.searchResults.length > 0 &&
                    <div className="list-container">
                        <ul className="list">
                            {this.state.searchResults.map(record => {
                                return (
                                    <li className="item" key={record} onClick={() => this.searchResultClicked(record)}>
                                        <input className="disabled-click"
                                               type="checkbox"
                                               checked={this.props.selectedCountries.has(record)}
                                               id={record}
                                               readOnly={true}/>
                                        <label htmlFor={record} className="disabled-click item-text">{record}</label>
                                    </li>
                                )
                            })}
                        </ul>

                    </div>}
                </div>
            </OutsideAlerter>
        )
    }
}