import React, { Component } from 'react';

interface Props {
    onOutsideClicked: () => void;
}

/**
 * Component that alerts if you click outside of it
 */
export default class OutsideAlerter extends Component<Props> {

    protected wrapperRef: any;


    componentDidMount() {
        document.addEventListener('mousedown', this.handleClickOutside);
    }

    componentWillUnmount() {
        document.removeEventListener('mousedown', this.handleClickOutside);
    }

    /**
     * Set the wrapper ref
     */
    setWrapperRef = (node: any) => {
        this.wrapperRef = node;
    };

    /**
     * Alert if clicked on outside of element
     */
    handleClickOutside = (event: any) => {
        if (this.wrapperRef && !this.wrapperRef.contains(event.target)) {
            this.props.onOutsideClicked();
        }
    };

    render() {
        return <div ref={this.setWrapperRef}>{this.props.children}</div>;
    }
}

