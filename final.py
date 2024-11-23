import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objs as go
from prophet import Prophet
import numpy as np
import pytz
import plotly.express as px
from backtesting import BacktestDeltaNeutral
import threading
import logging
from kiteconnect import KiteTicker, KiteConnect

st.sidebar.header("Developed by Govind Pande")

# Update page selection to include "Delta Neutral Hedging"
page_select = st.sidebar.selectbox(
    "Choose Section",
    [
        "Project Overview",
        "Stock Visualizations",
        "Share Holders Visualization",
        "Compare Stocks",
        "Backtest",
        "Backtest Viz",
        "Price Prediction",
        "Bring your own data",
        "Weekly Volatility & ^INDIAVIX",
        "Weekly Volatility Prediction with Prophet",
        "NIFTY 500 Sentiment Dashboard",
        "Option Price Analysis",
        "Delta Neutral Hedging",  # Added new section here
        "Delta Neutral Backtest",
    ],
)


def mainn():
    # Existing sections remain unchanged
    # ...

    # Add your existing code for other pages here
    # ...

    # New section for Delta Neutral Hedging
    if page_select == "Delta Neutral Hedging":
        st.title("Delta-Neutral Trading Strategy Monitor")

        # Input fields for API key and access token
        api_key = st.text_input("API Key", key="api_key_dn")
        access_token = st.text_input("Access Token", key="access_token_dn")

        # Input for instrument tokens
        instrument_tokens_input = st.text_input(
            "Instrument Tokens (comma-separated)",
            value="738561,5633",
            key="instrument_tokens_dn",
        )

        # Buttons for connecting and disconnecting
        col1, col2 = st.columns(2)
        with col1:
            connect_button = st.button("Connect", key="connect_dn")
        with col2:
            disconnect_button = st.button("Disconnect", key="disconnect_dn")

        # Initialize session state variables if not already done
        if "tick_data_dn" not in st.session_state:
            st.session_state.tick_data_dn = []
        if "connected_dn" not in st.session_state:
            st.session_state.connected_dn = False
        if "kws_dn" not in st.session_state:
            st.session_state.kws_dn = None

        # Define functions for starting KiteTicker and handling callbacks
        def start_kite_ticker(api_key, access_token, instrument_tokens):
            kws = KiteTicker(api_key, access_token)

            def on_ticks(ws, ticks):
                # Callback to receive ticks.
                logging.info("Ticks: {}".format(ticks))
                # Update the tick_data in session state
                st.session_state.tick_data_dn = ticks

            def on_connect(ws, response):
                # Callback on successful connect.
                # Subscribe to the list of instrument_tokens
                ws.subscribe(instrument_tokens)
                # Set mode to full for all instruments
                ws.set_mode(ws.MODE_FULL, instrument_tokens)

            def on_close(ws, code, reason):
                # On connection close stop the main loop
                ws.stop()
                st.session_state.connected_dn = False

            # Assign the callbacks.
            kws.on_ticks = on_ticks
            kws.on_connect = on_connect
            kws.on_close = on_close

            # Store the KiteTicker instance in session state
            st.session_state.kws_dn = kws

            # Connect to the websocket (non-blocking call)
            kws.connect(threaded=True)

        if connect_button and not st.session_state.connected_dn:
            if not api_key or not access_token:
                st.error("Please enter both API Key and Access Token.")
            else:
                # Convert instrument tokens to a list of integers
                try:
                    instrument_tokens = [
                        int(token.strip())
                        for token in instrument_tokens_input.split(",")
                    ]
                    # Start KiteTicker in a new thread
                    threading.Thread(
                        target=start_kite_ticker,
                        args=(api_key, access_token, instrument_tokens),
                        daemon=True,
                    ).start()
                    st.success("Connecting to KiteTicker...")
                    st.session_state.connected_dn = True
                except ValueError:
                    st.error("Please enter valid instrument tokens.")

        if disconnect_button and st.session_state.connected_dn:
            if st.session_state.kws_dn:
                st.session_state.kws_dn.close()
                st.success("Disconnected from KiteTicker.")
            st.session_state.connected_dn = False

        # Display tick data
        st.subheader("Live Tick Data")
        tick_display = st.empty()

        if st.session_state.connected_dn:
            if st.session_state.tick_data_dn:
                # Display the latest tick data
                tick_display.write(st.session_state.tick_data_dn)
            else:
                tick_display.write("Waiting for tick data...")
            # Auto-refresh the app every few seconds
            st.experimental_rerun()
        else:
            tick_display.write("Not connected.")

        if st.session_state.connected_dn:
            st.subheader("Place an Order")
            trading_symbol = st.text_input("Trading Symbol", key="trading_symbol_dn")
            transaction_type = st.selectbox(
                "Transaction Type", ["BUY", "SELL"], key="transaction_type_dn"
            )
            quantity = st.number_input(
                "Quantity", min_value=1, value=1, key="quantity_dn"
            )
            place_order_button = st.button("Place Order", key="place_order_dn")

            if place_order_button:
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(access_token)
                try:
                    order_id = kite.place_order(
                        variety=kite.VARIETY_REGULAR,
                        exchange=kite.EXCHANGE_NSE,  # Assuming NSE exchange
                        tradingsymbol=trading_symbol,
                        transaction_type=transaction_type,
                        quantity=int(quantity),
                        product=kite.PRODUCT_MIS,
                        order_type=kite.ORDER_TYPE_MARKET,
                    )
                    st.success(f"Order placed successfully. Order ID: {order_id}")
                except Exception as e:
                    st.error(f"Error placing order: {e}")
        else:
            st.write("Connect to the KiteTicker to place orders.")

    if page_select == "Delta Neutral Backtest":
        st.title("Delta Neutral Strategy Backtest")
    
        # User inputs
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
        start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
        end_date = st.date_input("End Date", value=pd.to_datetime('2021-01-01'))
        option_type = st.selectbox("Option Type", ['call', 'put'])
        strike_offset = st.number_input("Strike Price Offset (%)", value=0.0)
        risk_free_rate = st.number_input("Risk-Free Rate (%)", value=1.0) / 100
        hedge_threshold = st.number_input("Hedge Threshold (Delta)", value=0.1)
        transaction_cost = st.number_input("Transaction Cost (%)", value=0.1) / 100
    
        if st.button("Run Backtest"):
            # Run the backtest
            bt = BacktestDeltaNeutral(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                option_type,
                strike_offset / 100,
                risk_free_rate,
                hedge_threshold,
                transaction_cost,
            )
            bt.get_data()
            bt.estimate_volatility()
            bt.run_backtest()
            
            # Display results
            st.line_chart(bt.results['PortfolioValue'])
            st.write("Final Portfolio Value: ${:.2f}".format(bt.results['PortfolioValue'].iloc[-1]))

    # Add your existing code for other pages here
    # ...


def main():
    mainn()


if __name__ == "__main__":
    main()
