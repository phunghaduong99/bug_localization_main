/*
 *************************************************************************
 * Copyright (c) 2013 Actuate Corporation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *  Actuate Corporation - initial API and implementation
 *  
 *************************************************************************
 */

package org.eclipse.birt.data.oda.mongodb.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.eclipse.birt.data.oda.mongodb.internal.impl.DriverUtil;
import org.eclipse.birt.data.oda.mongodb.nls.Messages;
import org.eclipse.datatools.connectivity.oda.IConnection;
import org.eclipse.datatools.connectivity.oda.IDriver;
import org.eclipse.datatools.connectivity.oda.LogConfiguration;
import org.eclipse.datatools.connectivity.oda.OdaException;
import org.eclipse.datatools.connectivity.oda.util.manifest.DataTypeMapping;
import org.eclipse.datatools.connectivity.oda.util.manifest.ExtensionManifest;
import org.eclipse.datatools.connectivity.oda.util.manifest.ManifestExplorer;

import com.mongodb.Mongo;
import com.mongodb.MongoClientOptions;
import com.mongodb.MongoClientOptions.Builder;
import com.mongodb.MongoOptions;
import com.mongodb.MongoURI;
import com.mongodb.ReadPreference;
import com.mongodb.ServerAddress;

/**
 * Implementation class of IDriver for the MongoDB ODA runtime driver.
 */
public class MongoDBDriver implements IDriver
{
    public static final String ODA_DATA_SOURCE_ID = "org.eclipse.birt.data.oda.mongodb";  //$NON-NLS-1$

    private static final String MONGO_PROP_PREFIX = ""; //$NON-NLS-1$   // not using a prefix for now

    public static final String IGNORE_URI_PROP = MONGO_PROP_PREFIX.concat( "ignoreURI" ); //$NON-NLS-1$
    public static final String MONGO_URI_PROP = MONGO_PROP_PREFIX.concat( "mongoURI" ); //$NON-NLS-1$

    public static final String SERVER_HOST_PROP = MONGO_PROP_PREFIX.concat( "serverHost" ); //$NON-NLS-1$
    public static final String SERVER_PORT_PROP = MONGO_PROP_PREFIX.concat( "serverPort" ); //$NON-NLS-1$
    public static final String DBNAME_PROP = MONGO_PROP_PREFIX.concat( "databaseName" ); //$NON-NLS-1$
    public static final String USERNAME_PROP = MONGO_PROP_PREFIX.concat( "userName" ); //$NON-NLS-1$
    public static final String PASSWORD_PROP = MONGO_PROP_PREFIX.concat( "password" ); //$NON-NLS-1$

    // supported MongoOptions that are not covered in MongoURI
    public static final String SOCKET_KEEP_ALIVE_PROP = MONGO_PROP_PREFIX.concat( "socketKeepAlive" ); //$NON-NLS-1$
    // additional MongoDB ODA connection properties
    public static final String REQUEST_SESSION_PROP = MONGO_PROP_PREFIX.concat( "useRequestSession" ); //$NON-NLS-1$

    private static final MongoDBDriver sm_factory = new MongoDBDriver();
    private static final MongoOptions sm_defaultClientOptions = createDefaultClientOptions();

    private static ConcurrentMap<ServerNodeKey, Mongo> sm_mongoServerNodes;
    
    private static ConcurrentMap<ServerNodeKey, Mongo> getMongoServerNodes()
    {
        // different from Mongo.Holder (which uses MongoURI as key), 
        // this uses cached key based on a
        // MongoURI plus supported options not definable in MongoURI
        if( sm_mongoServerNodes == null )
        {
            synchronized( MongoDBDriver.class )
            {
                if( sm_mongoServerNodes == null )
                    sm_mongoServerNodes = new ConcurrentHashMap<ServerNodeKey, Mongo>();
            }
        }
        return sm_mongoServerNodes;
    }

    public static void close()
    {
        synchronized( MongoDBDriver.class )
        {
            if( sm_mongoServerNodes == null )
                return;

            for( Mongo node : sm_mongoServerNodes.values() )
            {
                node.close();
            }
            sm_mongoServerNodes.clear();
            sm_mongoServerNodes = null;
        }
    }

    private static Mongo getMongoNodeInstance( ServerNodeKey serverNodeKey )
        throws OdaException
    {
        // first check if a cached node already exists and reuse
        Mongo mongoNode = getMongoServerNodes().get( serverNodeKey );
        if( mongoNode != null )
            return mongoNode;
        
        // now try get mongo node based on server host/port and supported options
        mongoNode = createMongoNode( serverNodeKey );
        Mongo existingNode = getMongoServerNodes().putIfAbsent( serverNodeKey, mongoNode );  // cache the new mongo server node

        if( existingNode == null )  // the new one got in
            return mongoNode;

        // there was a race, and the new node lost;
        // close the new node, and return the existing one
        mongoNode.close();
        return existingNode;
    }
    
    static Mongo getMongoNode( Properties connProperties )
        throws OdaException
    {
        ServerNodeKey nodeKey = createServerNodeKey( connProperties );
        return getMongoNodeInstance( nodeKey );
    }
    
	/*
	 * @see org.eclipse.datatools.connectivity.oda.IDriver#getConnection(java.lang.String)
	 */
	public IConnection getConnection( String dataSourceType ) throws OdaException
	{
        // driver supports only one type of data source,
        // ignores the specified dataSourceType
        return new MDbConnection();
	}

	/*
	 * @see org.eclipse.datatools.connectivity.oda.IDriver#setLogConfiguration(org.eclipse.datatools.connectivity.oda.LogConfiguration)
	 */
	public void setLogConfiguration( LogConfiguration logConfig ) throws OdaException
	{
		// not supported
	}

	/*
	 * @see org.eclipse.datatools.connectivity.oda.IDriver#getMaxConnections()
	 */
    public int getMaxConnections() throws OdaException
    {
        // use default value defined in MongoClientOptions.connectionsPerHost;
        // this may be called before opening a connection, i.e. no instance of MongoOptions is available
        return 100;
    }
	
	/*
	 * @see org.eclipse.datatools.connectivity.oda.IDriver#setAppContext(java.lang.Object)
	 */
	public void setAppContext( Object context ) throws OdaException
	{
	    // do nothing; no support for pass-through context
	}

    /**
     * Returns the object that represents this extension's manifest.
     * @throws OdaException
     */
    static ExtensionManifest getManifest()
        throws OdaException
    {
        return ManifestExplorer.getInstance()
                .getExtensionManifest( ODA_DATA_SOURCE_ID );
    }
    
    /**
     * Returns the native data type name of the specified code, as
     * defined in this data source extension's manifest.
     * @param nativeTypeCode    the native data type code
     * @return                  corresponding native data type name
     * @throws OdaException     if lookup fails
     */
    static String getNativeDataTypeName( int nativeDataTypeCode ) 
        throws OdaException
    {
        DataTypeMapping typeMapping = 
                            getManifest().getDataSetType( null )
                                .getDataTypeMapping( nativeDataTypeCode );
        if( typeMapping != null )
            return typeMapping.getNativeType();
        return Messages.mDbDriver_nonDefinedDataType; 
    }

    /**
     * An enum of MongoDB ReadPreference that supports
     * localization of its display names.
     */
    public enum ReadPreferenceChoice
    {
        PRIMARY,
        PRIMARY_PREFERRED,
        SECONDARY,
        SECONDARY_PREFERRED,
        NEAREST;
        
        private ReadPreferenceChoice() {}
        
        public static ReadPreferenceChoice DEFAULT = PRIMARY;
        public static ReadPreference DEFAULT_PREFERENCE = ReadPreference.primary();

        public static ReadPreference getMongoReadPreference( String readPrefChoiceLiteral )
        {
            if( readPrefChoiceLiteral == null || readPrefChoiceLiteral.trim().isEmpty() )
                return null;  // use MongoDB default

            try
            {
                return ReadPreference.valueOf( readPrefChoiceLiteral.trim() );
            }
            catch( IllegalArgumentException ex )
            {
                // ignore, falls back to return default
            }
            return DEFAULT_PREFERENCE;  // default
        }

        public static ReadPreferenceChoice getReadPreferenceChoice( ReadPreference readPref )
        {
            if( readPref == null )
                return PRIMARY;     // default
           String readPrefName = readPref.getName();
            if( readPrefName == ReadPreference.primary().getName() )
                return PRIMARY;
            if( readPrefName == ReadPreference.primaryPreferred().getName() )
                return PRIMARY_PREFERRED;
            if( readPrefName == ReadPreference.secondary().getName() )
                return SECONDARY;
            if( readPrefName == ReadPreference.secondaryPreferred().getName() )
                return SECONDARY_PREFERRED;
            if( readPrefName == ReadPreference.nearest().getName() )
                return NEAREST;
            return PRIMARY;     // default
        }

        public String displayName()
        {
            // externalizes name, which is not provided by Mongo Java driver
            if( this == PRIMARY )
                return Messages.mDbDriver_readPrefPrimary; // ReadPreference.primary().getName();
            if( this == PRIMARY_PREFERRED )
                return Messages.mDbDriver_readPrefPrimaryPreferred; // ReadPreference.primaryPreferred().getName();
            if( this == SECONDARY )
                return Messages.mDbDriver_readPrefSecondary; // ReadPreference.secondary().getName();
            if( this == SECONDARY_PREFERRED )
                return Messages.mDbDriver_readPrefSecondaryPreferred; // ReadPreference.secondaryPreferred().getName();
            if( this == NEAREST )
                return Messages.mDbDriver_readPrefNearest; // ReadPreference.nearest().getName();
            return Messages.mDbDriver_readPrefPrimary;  // default
        }        
    }

    private static Mongo createMongoNode( ServerNodeKey serverNodeKey )
        throws OdaException
    {
        /* NOTE: Not able to use the new 2.10 MongoClient classes directly,
         *  as their public methods, as of 2.10.1, do not allow merging options of
         *  MongoClientURL and MongoClientOptions
         */
        List<String> hosts = serverNodeKey.getServerHosts();
        String standaloneHost = hosts.size() == 1 ? 
                            hosts.get(0) : null;
        Integer port = serverNodeKey.getServerPort();
        MongoOptions options = serverNodeKey.getOptions();
        if( options == null )
            options = sm_defaultClientOptions;

        try
        {
            if( standaloneHost != null )
            {
                ServerAddress serverAddr = port != null ?
                        new ServerAddress( standaloneHost, port ) :
                        new ServerAddress( standaloneHost );
                return new Mongo( serverAddr, options );
            }
            else
            {
                List<ServerAddress> serverSeeds = new ArrayList<ServerAddress>( hosts.size() );
                for( String host : hosts )
                    serverSeeds.add( new ServerAddress( host ) );
                return new Mongo( serverSeeds, options );
            }
        }
        catch( Exception ex )
        {
            throw new OdaException( ex );
        }
    }
 
    private static ServerNodeKey createServerNodeKey( Properties connProperties )
    {
        return sm_factory.new ServerNodeKey( connProperties );
    }

    private class ServerNodeKey
    {
        private List<String> m_serverHosts;
        private Integer m_serverPort;
        private String m_userName;
        private MongoOptions m_options;
        
        ServerNodeKey( Properties connProperties )
        {
            // first check if user-defined URL exists, which takes precedence 
            // if not flagged to ignore by the ignoreURI property
            MongoURI mongoUri = getMongoURI( connProperties );
            MongoOptions nodeOptions = getMongoOptions( mongoUri, connProperties );

            if( mongoUri != null )  // has user-defined MongoURI
            {
                init( mongoUri, nodeOptions );
                return;
            }
            
            // none or invalid user-defined URL, initialize from individual properties
            init( connProperties, nodeOptions );
        }
        
        private void init( MongoURI mongoUri, MongoOptions options )
        {
            m_serverHosts = new ArrayList<String>(mongoUri.getHosts().size());
            for( String host : mongoUri.getHosts() )
                m_serverHosts.add( host );   // may contain optional ":<port>"
            m_serverPort = ServerAddress.defaultPort();
            m_userName = mongoUri.getUsername();            
            m_options = options;
            
            // trace logging
            logInitializedValues( "ServerNodeKey#init(MongoURI,MongoOptions)" ); //$NON-NLS-1$
        }

        private void init( Properties connProperties, MongoOptions options )
        {
            m_serverHosts = new ArrayList<String>(1);
            m_serverHosts.add( getStringPropValue( connProperties, SERVER_HOST_PROP ) );
            m_serverPort = getIntegerPropValue( connProperties, SERVER_PORT_PROP );
            m_userName = getUserName( connProperties );            
            m_options = options;
            
            // trace logging
            logInitializedValues( "ServerNodeKey#init(Properties,MongoOptions)" ); //$NON-NLS-1$
        }
        
        private void logInitializedValues( String methodName )
        {
            if( getLogger().isLoggable( Level.FINEST ) )
                getLogger().finest( Messages.bind( "{0}: hosts= {1}, port= {2}, user= {3}, options= {4}", //$NON-NLS-1$
                        new Object[]{ methodName, m_serverHosts, m_serverPort, m_userName, m_options} ));          
        }

        @Override
        public boolean equals( Object obj )
        {
            if( super.equals( obj ) )
                return true;
            if( !(obj instanceof ServerNodeKey) )
                return false;
            
            // compare the attribute values
            ServerNodeKey thatKey = (ServerNodeKey)obj;
            if( this.m_serverHosts == null && thatKey.m_serverHosts != null )
                return false;
            if( this.m_serverHosts != null && 
                    ! this.m_serverHosts.equals( thatKey.m_serverHosts ) )
                return false;

            if( this.m_serverPort == null && thatKey.m_serverPort != null )
                return false;
            if( this.m_serverPort != null && 
                    ! this.m_serverPort.equals( thatKey.m_serverPort ) )
                return false;

            if( this.m_options == null && thatKey.m_options != null )
                return false;
            if( this.m_options != null && 
                    ! this.m_options.toString().equals( thatKey.m_options.toString() ) )
                return false;

            if( this.m_userName == null && thatKey.m_userName != null )
                return false;
            if( this.m_userName != null && 
                    ! this.m_userName.equals( thatKey.m_userName ) )
                return false;

            return true;
        }

        @Override
        public int hashCode()
        {
            // use its attributes for hashcode if exists
            int hashCode = 0;
            if( m_serverHosts != null )
                hashCode = m_serverHosts.hashCode();
            
            if( m_serverPort != null )
                hashCode = hashCode ^ m_serverPort.hashCode();
            
            if( m_options != null )
                hashCode = hashCode ^ m_options.toString().hashCode();
            
            if( m_userName != null )
                hashCode = hashCode ^ m_userName.hashCode();
            
            return hashCode == 0 ? super.hashCode() : hashCode;
        }

        private List<String> getServerHosts()
        {
            if( m_serverHosts == null )
                return Collections.emptyList();
            return m_serverHosts;
        }

        private Integer getServerPort()
        {
            return m_serverPort;
        }

        private MongoOptions getOptions()
        {
            return m_options;
        }

        @SuppressWarnings("unused")
        private String getUsername()
        {
            return m_userName;
        }
    }

    /*
     * Utility methods to handle mongo options, adopting the defaults set by 
     * the MongoClientOptions.
     * NOTE: Not able to use MongoClientOptions directly, 
     * as its public API, as of 2.10.1, does not allow merging option values from a MongoClientURI.
     */
    
    private static MongoOptions getMongoOptions( MongoURI mongoUri, Properties connProperties )
    {
        // check if additinal options exists in connection properties 
        // that are not covered by the mongoURI
        MongoOptions nodeOptions = mongoUri != null ? getURIOptions( mongoUri ) : null;                
        return addSupplementalOptions( nodeOptions, connProperties );
    }

    private static MongoOptions createDefaultClientOptions()
    {
        Builder optionsBuilder = new MongoClientOptions.Builder();
        return new MongoOptions( optionsBuilder.build() );
    }

    private static MongoOptions getURIOptions( MongoURI mongoUri )
    {
        MongoOptions uriOptions = mongoUri.getOptions();

        // explicitly adopts those default client options, which were overridden
        // by MongoClientOptions.Builder#legacyDefaults()
        uriOptions.setConnectionsPerHost( sm_defaultClientOptions.getConnectionsPerHost() );
        uriOptions.setWriteConcern( sm_defaultClientOptions.getWriteConcern() );
        
        return uriOptions;
    }

    private static MongoOptions addSupplementalOptions( MongoOptions mongoOptions, Properties connProperties )
    {
        if( hasKeepSocketAlive( connProperties ) )   // need to change setting, as MongoDB default is false
        {
            if( mongoOptions == null )
                mongoOptions = createDefaultClientOptions();
            mongoOptions.setSocketKeepAlive( true );            
        }

        return mongoOptions;        
    }

    private static Boolean hasKeepSocketAlive( Properties connProperties )
    {
        String keepSocketAlivePropValue = getStringPropValue( connProperties, SOCKET_KEEP_ALIVE_PROP );
        if( keepSocketAlivePropValue == null )    // supported option is not defined
            return Boolean.FALSE;
        boolean keepSocketAlive = Boolean.valueOf( keepSocketAlivePropValue );
        if( keepSocketAlive == false )   // mongoDB default
            return Boolean.FALSE;    // using default value, no need to return value
        
        return Boolean.TRUE;
    }

    static String getDatabaseName( Properties connProps )
    {
        MongoURI mongoURI = getMongoURI( connProps );
        if( mongoURI != null )
            return mongoURI.getDatabase();
        
        // no mongoURI specified, get from the individual property
        return getStringPropValue( connProps, DBNAME_PROP );
    }

    static String getUserName( Properties connProps )
    {
        return getStringPropValue( connProps, USERNAME_PROP );
    }

    static String getPassword( Properties connProps )
    {
        return getStringPropValue( connProps, PASSWORD_PROP );
    }

    private static MongoURI getMongoURI( Properties connProps )
    {
        // check if explicitly indicated not to use URI, even if URI value exists
        Boolean ignoreURI = getBooleanPropValue( connProps, IGNORE_URI_PROP );
        if( ignoreURI != null && ignoreURI )
            return null;

        String uri = getStringPropValue( connProps, MONGO_URI_PROP );
        if( uri == null || uri.isEmpty() )
            return null;
        
        try
        {
            return new MongoURI( uri );
        }
        catch( Exception ex )
        {
            // log and ignore
            getLogger().log( Level.INFO, Messages.bind( "Invalid Mongo Database URI: {0}", uri ), ex ); //$NON-NLS-1$
        }
        return null;
    }

    /*
     * Not currently used.
     */
    @SuppressWarnings("unused")
    private static String formatMongoURI( Properties connProps )
    {
        // format a Mongo URI text from supported connection properties
        // Mongo URI syntax:
        // mongodb://[username:password@]host1[:port1]...[,hostN[:portN]][/[database][?options]]

        // validate that the mininum required URI part exists
        String serverHost = getStringPropValue( connProps, SERVER_HOST_PROP );
        if( serverHost == null || serverHost.isEmpty() )
            throw new IllegalArgumentException( Messages.mDbDriver_missingValueServerHost );
        
        StringBuffer buf = new StringBuffer( MongoURI.MONGODB_PREFIX );

        String username = getUserName( connProps );
        if( username != null && ! username.isEmpty() )
        {
            String passwd = getPassword( connProps );
            buf.append( username );
            buf.append( ':' );
            buf.append( passwd );
            buf.append( '@' );
        }

        buf.append( serverHost );

        Integer serverPort = getIntegerPropValue( connProps, SERVER_PORT_PROP );
        if( serverPort != null )
        {
            buf.append( ':' );
            buf.append( serverPort );
        }

        String dbName = getStringPropValue( connProps, DBNAME_PROP );
        if( dbName != null )
        {
            buf.append( '/' );
            buf.append( dbName );
        }

        //  comment out inclusion of all options in the generated URI text, cuz
        //  the MongoURI parser does not yet support all the options allowed in MongoOptions
/*
        MongoOptions options = createMongoOptions( connProps );
        if( options != null )
        {
            // if database is absent, a '/' is still required
            if( dbName == null )
                buf.append( '/' );
            buf.append( '?' );

            String optionsText = options.toString();
            // replace the option separator ',' with ';' as required in MongoURI
            optionsText = optionsText.replace( ',', ';' );
            buf.append( optionsText );
        }
*/        
        return buf.toString();
    }

    static String getStringPropValue( Properties props, String propName )
    {
        String propValue = props.getProperty( propName );
        return propValue != null ? propValue.trim() : null;
    }

    static Boolean getBooleanPropValue( Properties props, String propName )
    {
        String propValue = getStringPropValue( props, propName );
        if( propValue == null || propValue.isEmpty() )
            return null;
        return Boolean.valueOf( propValue );
    }

    static Integer getIntegerPropValue( Properties props, String propName )
    {
        String propValue = getStringPropValue( props, propName );
        if( propValue == null || propValue.isEmpty() )
            return null;
        try
        {
            return Integer.valueOf( propValue );
        }
        catch( NumberFormatException ex )
        {
            // log and ignore
            getLogger().log( Level.INFO, "MongoDBDriver#getIntegerPropValue ignoring exception: " + ex ); //$NON-NLS-1$
        }
        return null;
    }

    static Logger getLogger()
    {
        return DriverUtil.getLogger();
    }

}
