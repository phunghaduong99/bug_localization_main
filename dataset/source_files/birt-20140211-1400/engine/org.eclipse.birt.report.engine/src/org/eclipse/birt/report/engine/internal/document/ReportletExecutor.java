/*******************************************************************************
 * Copyright (c) 2004 Actuate Corporation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *  Actuate Corporation  - initial API and implementation
 *******************************************************************************/

package org.eclipse.birt.report.engine.internal.document;

import java.io.IOException;

import org.eclipse.birt.core.exception.BirtException;
import org.eclipse.birt.report.engine.api.EngineException;
import org.eclipse.birt.report.engine.executor.ExecutionContext;
import org.eclipse.birt.report.engine.i18n.MessageConstants;
import org.eclipse.birt.report.engine.internal.document.v3.ReportletExecutorV3;
import org.eclipse.birt.report.engine.internal.document.v4.ReportletExecutorV4;

public class ReportletExecutor extends ReportExecutorWrapper
{

	public ReportletExecutor( ExecutionContext context, long offset )
			throws BirtException
	{
		try
		{
			int version = getVersion( context.getReportDocument( ) );
			switch ( version )
			{
				case EXECUTOR_VERSION_3 :
					executor = new ReportletExecutorV3( context, offset );
					break;
				case EXECUTOR_VERSION_4 :
					executor = new ReportletExecutorV4( context, offset );
					break;
				default :
					throw new EngineException( MessageConstants.UNSUPPORTED_DOCUMENT_VERSION_ERROR
							, version );
			}
		}
		catch ( IOException ex )
		{
			throw new EngineException( ex.getLocalizedMessage( ), ex );
		}
	}
}
